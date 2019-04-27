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
from deepflow.optimizers import MALA, pSGLD, pSGLDmod
from deepflow.mrst_coupling import PytorchMRSTCoupler, load_production_data, load_gradients
from deepflow.storage import create_dataset
from deepflow.utils import set_seed, load_generator, report_latent_vector_stats, print_header

import time
import random 
import gc
import os 
import argparse 
import sys
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)

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
    parser.add_argument("--beta1", type=float, default=0.9, help="Momentum first parameter") 
    parser.add_argument("--beta2", type=float, default=0.999, help="Momentum second parameter") 
    parser.add_argument("--iterations", type=int, default=200, help="Number of gradient steps")
    parser.add_argument("--optimize_wells", action="store_true", help="Match wells")
    parser.add_argument("--optimize_flow", action="store_true", help="Match flow behavior")
    parser.add_argument("--use_prior_loss", action="store_true", help="Regularize latent variables to be Gaussian. Same as weight decay but uses pytorch distributions. Set Weight Decay to 0!")
    parser.add_argument("--use_kl_loss", action="store_true", help="Regularize latent variables to be Gaussian. Same as weight decay but uses pytorch distributions. Set Weight Decay to 0!")
    parser.add_argument("--unconditional", action="store_true", help="Do only one forward pass")
    parser.add_argument('--well_locations', nargs='+', type=int, default=[8, 120])
    parser.add_argument("--early_stopping", action="store_true", help="Stop early (only used for well only optimisation)")
    parser.add_argument("--wells_only", action="store_true", help="Optimize wells only.")
    parser.add_argument("--target_accuracy", type=float, default=1.0, help="Early stopping criterion for well only optimisation")
    parser.add_argument("--gamma", type=float, default=0.99, help="Learning-Rate Scheduler Gamma") 
    logger.info('Parsing CMD Line Arguments')
    args = parser.parse_args(argv)
    logger.info('Completed Parsing CMD Line Arguments')
    return args

def compute_well_loss(i, x, x_gt, well_locations, alpha=1.):
    x_prob = x[0, well_locations, :].view(1, 1, len(well_locations), 64)
    x_gt_indicator = x_gt[0, well_locations, :].view(1, 1, len(well_locations), 64)

    loss_f = nn.BCELoss(reduction="sum")
    
    loss = alpha*loss_f(x_prob, x_gt_indicator)
    acc = accuracy_score(
                        x_gt_indicator.cpu().detach().numpy().astype(int).flatten(),
                        np.where(x_prob.cpu().detach().numpy().flatten() > 0.5, 1, 0)
                        )
    return loss, acc

def compute_prior_loss_kl_divergence(z, alpha=1.):
    z_dist = torch.randn_like(z.view(1, 100))
    z_dist_mean = z_dist.mean()
    z_dist_std = z_dist.std()

    z_mean = z.mean()
    z_std = z.std()
    
    z1m = z_mean
    z2m = z_dist_mean
    
    z1s = z_std
    z2s = z_dist_std
    
    prior_loss = alpha*torch.log(z2s/z1s)+(z1s**2+(z1m-z2m)**2)/(2*z2s**2)-0.5
    return prior_loss

def compute_prior_loss(z, alpha=1.):
    pdf = torch.distributions.Normal(0, 1)
    logProb = pdf.log_prob(z.view(1, 100)).mean(dim=1)
    prior_loss = -alpha*logProb.mean()
    return prior_loss

def optimize(args, z, generator, optimizer, stepper):
    z_prior = z.clone()
    
    logger.info('Setup Paths')
    working_dir = os.path.expandvars(args.working_dir)
    output_path = os.path.expandvars(args.output_dir)
    matlab_path = os.path.join(working_dir, args.matlab_dir)
    mrst_path = os.path.join(working_dir, args.mrst_dir)

    well_loss = torch.from_numpy(np.array([-999.]))
    flow_loss = torch.from_numpy(np.array([-999.]))
    prior_loss = torch.from_numpy(np.array([-999.]))
    well_acc = -999.

    matlab_command =  ["matlab", "-nodisplay", "-nosplash", "-nodesktop", "-r"]
    fcall = ['run("'+os.path.join(mrst_path, "startup.m")+'"), run("'+os.path.join(matlab_path, "run_adjoint.m")+'"), exit']
    fcall_full = ['run("'+os.path.join(mrst_path, "startup.m")+'"), run("'+os.path.join(matlab_path, "run_adjoint_full.m")+'"), exit']

    external_commands = {"command": matlab_command, "call": fcall, "matlab_path": matlab_path}
    external_commands_full = {"command": matlab_command, "call": fcall_full, "matlab_path": matlab_path}

    ref_fname = os.path.join(matlab_path, "utils/vertcase3_noise/ws_ref.mat")
    syn_fname = os.path.join(matlab_path, "utils/synthetic/ws.mat")
    grad_name = os.path.join(matlab_path, "utils/synthetic/grad.mat")

    logger.info('Load Reference Case Data')
    ref_data = load_production_data(ref_fname, "ws_ref")
    np.save(os.path.join(output_path, "prod_data_ref.npy"), ref_data)
     
    logger.info('Load Reference Geological Model')
    x_gt = np.load(os.path.join(working_dir, args.reference_model))
    x_gt = torch.from_numpy(x_gt).float()

    logger.info('Starting Optimization Loop')
    for i in range(args.iterations):
        logger.info('Started Iteration %1.2i'%i)
        
        logger.info('Reset Latent Variable Gradients')
        optimizer.zero_grad()

        logger.info('Forward Pass GAN Generator Iteration %1.2i'%i)
        k, poro, x = generator(z)
        
        logger.info('Computing Well Loss')
        well_loss, well_acc = compute_well_loss(i, x, x_gt, args.well_locations)
        logger.info('[Well Loss]: %1.3f [Well Accuracy]: %1.2f' % (well_loss.item(), well_acc))

        if args.optimize_wells:
            logger.info('Using Well Loss, Performing Backward')
            well_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(z, 5.0)
        
        logger.info('Computing Gaussian Prior Loss')
        prior_loss_l2 = compute_prior_loss(z, alpha=1.)
        logger.info('[Gaussian Prior Loss]: %1.3f '%prior_loss_l2.item())

        logger.info('Computing KL-Divergence Loss')
        prior_loss_kl = compute_prior_loss_kl_divergence(z, alpha=1.)        
        logger.info('[KL-Divergence Loss]: %1.3f '%prior_loss_kl.item())

        if args.use_kl_loss:
            logger.info('Using KL-Divergence Loss Loss, Performing Backward')
            prior_loss_kl.backward()
        elif args.use_prior_loss:
            logger.info('Using Gaussian Prior Loss, Performing Backward')
            prior_loss_l2.backward() 

        if args.optimize_flow or args.optimize_wells or args.unconditional:
            logger.info('Using Flow Loss, Performing Forward Pass')
            coupler = PytorchMRSTCoupler()
            layer = coupler.apply
            
            flow_loss = layer(k, poro, external_commands).float()
            logger.info('[Flow Loss]: %1.3f '%flow_loss.item())

            if not args.wells_only:
                flow_loss.backward(retain_graph=True)
            
            logger.info('Loading Gradients and Production History')
            grads = load_gradients(grad_name)
            syn_data = load_production_data(syn_fname, "ws")
            
            logger.info('Simulate Full Production History')
            flow_loss_full = layer(k, poro, external_commands_full).float()
            syn_data_full = load_production_data(syn_fname, "ws")
            logger.info('Completed Flow Loss')
   
        if not args.unconditional:
            logger.info('Performing Gradient Descent')
            optimizer.step()
            if stepper:
                logger.info('Reducing Step Size')
                stepper.step()

        logger.info('Storing Iteration Output')
        ds = create_dataset(syn_data, syn_data_full,
                            np.array([poro.detach().numpy()[0, 0].T, k.detach().numpy()[0, 0].T]), 
                            grads, 
                            z.view(1, 50, 2, 1).detach().numpy(),
                            z_prior.view(1, 50, 2, 1).detach().numpy(),
                            z.grad.view(1, 50, 2, 1).numpy(),
                            np.array([[flow_loss.item(), well_loss.item(), well_acc, prior_loss_l2.item(), prior_loss_kl.item()]]))

        ds.to_netcdf(os.path.join(output_path, "iteration_"+str(i)+".nc"))
        logger.info('Completed Iteration Output')
        
        if args.early_stopping:
            if well_acc >= args.target_accuracy and args.early_stopping:
                logger.info('Early Stopping on iteration: %1.2i' % i)
                break
        
        if args.unconditional:
            break
        logger.info('Completed Iteration %1.2i'%i)

    return None

def main(args):
    logger.info('Starting DeepFlow')
    logger.info('')
    print_header()
    logger.info('')
    logger.info('Setting Random Seed: %1.2i' % args.seed)
    set_seed(args.seed)
    working_dir = os.path.expandvars(args.working_dir)
    checkpoints_path = os.path.join(working_dir, args.checkpoints_dir, "generator_facies_multichannel_4_6790.pth")

    logger.info('Inititalizing GAN Generator')
    generator = load_generator(checkpoints_path)
    
    logger.info('Sampling from Latent Space')
    z = torch.randn(1, 50, 1, 2)
    z.requires_grad = True

    stepper = None
    logger.info('Setting up optimizer '+args.optimizer)
    if args.optimizer == "sgd":
        optimizer = SGD([z], lr=args.lr, momentum=args.beta1, weight_decay=args.weight_decay)
    elif args.optimizer =="rmsprop":
        optimizer = RMSprop([z], lr=args.lr, momentum=args.beta1, weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        optimizer = Adam([z], lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    elif args.optimizer == "psgld":
        optimizer = pSGLDmod([z], lr=args.lr) #betas=(args.beta1, args.beta2), , weight_decay=args.weight_decay
    elif args.optimizer == "mala":
        optimizer = MALA(params=[z], lr=args.lr, weight_decay=args.weight_decay, eps3=args.weight_decay)
        stepper = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    
    states = optimize(args, z, generator, optimizer, stepper=stepper)

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
