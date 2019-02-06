import torch 
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import SGD, Adam, RMSprop, Rprop
import torch.nn.functional as functional

import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from deepflow.generator import PermeabilityGeneratorMRST as PermeabilityGenerator
from deepflow.networks import GeneratorMultiChannel
from deepflow.optimizers import MALA
from deepflow.mrst_coupling import PytorchMRSTCoupler, load_production_data, load_gradients
from deepflow.storage import create_dataset
from deepflow.utils import set_seed, load_generator, report_latent_vector_stats

import time
import random 
import gc
import os 
import argparse 
import sys

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--working_dir", type=str, default="./", help="Working directory")
    parser.add_argument("--output_dir", type=str, default="output_mrst", help="Output directory")
    parser.add_argument("--matlab_dir", type=str, default="./mrst/mrst-2018a/modules/optimization/examples/model2Dtest/", help="Matlab files directory") 
    parser.add_argument("--mrst_dir", type=str, default="./mrst/mrst-2018a", help="Matlab files directory") 
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints", help="Checkpoints directory")  
    parser.add_argument("--reference_model", type=str, default="reference/model_67_x.npy", help="Reference Model")   
    parser.add_argument("--optimizer", type=str, default="sgd", help="Optimization Method")
    parser.add_argument("--lr", type=float, default=1e-1, help="Learning Rate")   
    parser.add_argument("--seed", type=int, default=0, help="Random Seed") 
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Momentum first parameter") 
    parser.add_argument("--beta1", type=float, default=0.5, help="Momentum first parameter") 
    parser.add_argument("--beta2", type=float, default=0.9, help="Momentum second parameter") 
    parser.add_argument("--iterations", type=int, default=200, help="Number of gradient steps")
    parser.add_argument("--optimize_wells", action="store_true", help="Match wells")
    parser.add_argument("--optimize_flow", action="store_true", help="Match flow behavior")
    parser.add_argument('--well_locations', nargs='+', type=int, default=[8, 120])
    parser.add_argument("--early_stopping", action="store_true", help="Stop early (only used for well only optimisation)")
    parser.add_argument("--target_accuracy", type=float, default=0.9, help="Early stopping criterion for well only optimisation")
    args = parser.parse_args(argv)
    return args

def compute_well_loss(i, x, x_gt, well_locations):
    x_prob = x[0, well_locations, :].view(1, 1, len(well_locations), 64)
    x_gt_indicator = x_gt[0, well_locations, :].view(1, 1, len(well_locations), 64)

    loss_f = nn.BCELoss(reduction='mean')
    
    loss = loss_f(x_prob, x_gt_indicator)
    acc = accuracy_score(
                        x_gt_indicator.cpu().detach().numpy().astype(int).flatten(),
                        np.where(x_prob.cpu().detach().numpy().flatten() > 0.5, 1, 0)
                        )
    return loss, acc

def optimize(args, z, generator, optimizer):
    working_dir = os.path.expandvars(args.working_dir)
    output_path = os.path.expandvars(args.output_dir)
    matlab_path = os.path.join(working_dir, args.matlab_dir)
    mrst_path = os.path.join(working_dir, args.mrst_dir)

    well_loss = torch.from_numpy(np.array([-999.]))
    flow_loss = torch.from_numpy(np.array([-999.]))
    well_acc = -999.

    matlab_command =  ["matlab", "-nodisplay", "-nosplash", "-nodesktop", "-r"]
    fcall = ['run("'+os.path.join(mrst_path, "startup.m")+'"), run("'+os.path.join(matlab_path, "run_adjoint.m")+'"), exit']

    external_commands = {"command": matlab_command, "call": fcall, "matlab_path": matlab_path}

    ref_fname = os.path.join(matlab_path, "utils/vertcase3/ws_ref.mat")
    syn_fname = os.path.join(matlab_path, "utils/synthetic/ws.mat")

    grad_name = os.path.join(matlab_path, "utils/synthetic/grad.mat")

    ref_data = load_production_data(ref_fname, "ws_ref")
    np.save(os.path.join(output_path, "prod_data_ref.npy"), ref_data)
     
    x_gt = np.load(os.path.join(working_dir, args.reference_model))
    x_gt = torch.from_numpy(x_gt).float()

    for i in range(args.iterations):
        optimizer.zero_grad()
        k, poro, x = generator(z)
        
        well_loss, well_acc = compute_well_loss(i, x, x_gt, args.well_locations)
        
        if args.optimize_wells:
            well_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(z, 5.0)
            print("Well BCE Loss: %2.3f" % well_loss.item(), "Wells BCE Accuracy: %1.2f" % well_acc)
            report_latent_vector_stats(i, z, "well loss")

        if args.optimize_flow or (args.optimize_wells and well_acc >= args.target_accuracy):
            coupler = PytorchMRSTCoupler()
            layer = coupler.apply

            flow_loss = layer(k, poro, external_commands).float()
            flow_loss.backward()
            print("Flow Loss: ", flow_loss.item())
            report_latent_vector_stats(i, z, "flow loss")
        
        optimizer.step()
        report_latent_vector_stats(i, z, "optimizer step")

        grads = load_gradients(grad_name)
        syn_data = load_production_data(syn_fname, "ws")
        ds = create_dataset(syn_data, 
                                np.array([poro.detach().numpy()[0, 0].T, k.detach().numpy()[0, 0].T]), 
                                grads, 
                                z.view(1, 50, 2, 1).detach().numpy(),
                                z.grad.view(1, 50, 2, 1).numpy(),
                                np.array([[flow_loss.item(), well_loss.item(), well_acc]]))

        ds.to_netcdf(os.path.join(output_path, "iteration_"+str(i)+".nc"))
        print("Stored Iteration Output")
        if args.optimize_wells:
            if well_acc >= args.target_accuracy and args.early_stopping:
                print("Early Stopping on iteration: ", i)
                break
        print("")

    return None

def main(args):
    print("Seed ", args.seed)
    set_seed(args.seed)
    working_dir = os.path.expandvars(args.working_dir)
    checkpoints_path = os.path.join(working_dir, args.checkpoints_dir, "generator_facies_multichannel_4_6790.pth")

    generator = load_generator(checkpoints_path)
    
    z = torch.randn(1, 50, 1, 2)
    z.requires_grad = True

    if args.optimizer == "sgd":
        optimizer = SGD([z], lr=args.lr, momentum=args.beta1, weight_decay=args.weight_decay)
    elif args.optimizer =="mala":
        optimizer = MALA([z], lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer =="rmsprop":
        optimizer = RMSprop([z], lr=args.lr, momentum=args.beta1, weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        optimizer = Adam([z], lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)

    states = optimize(args, z, generator, optimizer)

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
