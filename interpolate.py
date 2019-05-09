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
from deepflow.losses import compute_prior_loss, compute_prior_loss_kl_divergence, compute_well_loss

import xarray as xr

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
    parser.add_argument("--z_file_1", type=str, default="./", help="Working directory")
    parser.add_argument("--z_file_2", type=str, default="./", help="Working directory")
    parser.add_argument("--output_dir", type=str, default="output_mrst", help="Output directory")
    parser.add_argument("--matlab_dir", type=str, default="./mrst/mrst-2018a/modules/optimization/examples/model2Dtest/", help="Matlab files directory") 
    parser.add_argument("--mrst_dir", type=str, default="./mrst/mrst-2018a", help="Matlab files directory") 
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints", help="Checkpoints directory")  
    parser.add_argument("--reference_model", type=str, default="reference/model_67_x.npy", help="Reference Model")   
    parser.add_argument("--seed", type=int, default=0, help="Random Seed") 
    parser.add_argument("--iterations", type=int, default=200, help="Number of gradient steps")
    parser.add_argument("--optimize_wells", action="store_true", help="Match wells")
    parser.add_argument("--optimize_flow", action="store_true", help="Match flow behavior")
    parser.add_argument("--cluster", action="store_true", help="Run on Cluster")
    parser.add_argument("--use_prior_loss", action="store_true", help="Regularize latent variables to be Gaussian. Same as weight decay but uses pytorch distributions. Set Weight Decay to 0!")
    parser.add_argument("--use_kl_loss", action="store_true", help="Regularize latent variables to be Gaussian using an empirical KL-Divergence")
    parser.add_argument('--well_locations', nargs='+', type=int, default=[8, 120])
    parser.add_argument("--wells_only", action="store_true", help="Optimize wells only.")
    logger.info('Parsing CMD Line Arguments')
    args = parser.parse_args(argv)
    logger.info('Completed Parsing CMD Line Arguments')
    return args

def interpolate(args, zs, generator, output_path):
    z_prior = zs[0].clone()
    
    logger.info('Setup Paths')
    working_dir = os.path.expandvars(args.working_dir)
    
    matlab_path = os.path.join(working_dir, args.matlab_dir)
    mrst_path = os.path.join(working_dir, args.mrst_dir)

    well_loss = torch.from_numpy(np.array([-999.]))
    flow_loss = torch.from_numpy(np.array([-999.]))
    prior_loss = torch.from_numpy(np.array([-999.]))
    well_acc = -999.

    case_name = "vertcase3_noise"
    os.environ["case_name"] = case_name

    matlab_command =  ["matlab", "-nodisplay", "-nosplash", "-nodesktop", "-r"]
    fcall = ['run("'+os.path.join(mrst_path, "startup.m")+'"), run("'+os.path.join(matlab_path, "run_adjoint.m")+'"), exit']
    fcall_full = ['run("'+os.path.join(mrst_path, "startup.m")+'"), run("'+os.path.join(matlab_path, "run_adjoint_full.m")+'"), exit']

    external_commands = {"command": matlab_command, "call": fcall, "matlab_path": matlab_path}
    external_commands_full = {"command": matlab_command, "call": fcall_full, "matlab_path": matlab_path}

    ref_fname = os.path.join(matlab_path, "utils/"+case_name+"/ws_ref.mat")
    syn_fname = os.path.join(matlab_path, "utils/synthetic/ws.mat")
    grad_name = os.path.join(matlab_path, "utils/synthetic/grad.mat")

    logger.info('Load Reference Case Data')
    ref_data = load_production_data(ref_fname, "ws_ref")
    np.save(os.path.join(output_path, "prod_data_ref.npy"), ref_data)
     
    logger.info('Load Reference Geological Model')
    x_gt = np.load(os.path.join(working_dir, args.reference_model))
    x_gt = torch.from_numpy(x_gt).float()

    logger.info('Starting Optimization Loop')
    for i, z in enumerate(zs):
        logger.info('Started Iteration %1.2i'%i)

        logger.info('Forward Pass GAN Generator Iteration %1.2i'%i)
        k, poro, x = generator(z)
        
        logger.info('Computing Well Loss')
        well_loss, well_acc = compute_well_loss(i, x, x_gt, args.well_locations)
        logger.info('[Well Loss]: %1.3f [Well Accuracy]: %1.2f' % (well_loss.item(), well_acc))

        
        logger.info('Computing Gaussian Prior Loss')
        prior_loss_l2 = compute_prior_loss(z, alpha=1.)
        logger.info('[Gaussian Prior Loss]: %1.3f '%prior_loss_l2.item())

        logger.info('Computing KL-Divergence Loss')
        prior_loss_kl = compute_prior_loss_kl_divergence(z, alpha=1.)        
        logger.info('[KL-Divergence Loss]: %1.3f '%prior_loss_kl.item())

        logger.info('Using Flow Loss, Performing Forward Pass')
        coupler = PytorchMRSTCoupler()
        layer = coupler.apply
        
        flow_loss = layer(k, poro, external_commands).float()
        logger.info('[Flow Loss]: %1.3f '%flow_loss.item())
        
        logger.info('Loading Gradients and Production History')
        grads = load_gradients(grad_name)
        syn_data = load_production_data(syn_fname, "ws")

        logger.info('Storing Iteration Output')
        ds = create_dataset(syn_data, syn_data,
                            np.array([poro.detach().numpy()[0, 0].T, k.detach().numpy()[0, 0].T]), 
                            grads, 
                            z.view(1, 50, 2, 1).detach().numpy(),
                            z_prior.view(1, 50, 2, 1).detach().numpy(),
                            z_prior.view(1, 50, 2, 1).numpy(),
                            np.array([[flow_loss.item(), well_loss.item(), well_acc, prior_loss_l2.item(), prior_loss_kl.item()]]))

        ds.to_netcdf(os.path.join(output_path, "iteration_"+str(i)+".nc"))
        logger.info('Completed Iteration Output')

        logger.info('Completed Iteration %1.2i'%i)

    return None

def get_latent_vector(file):
    z = None
    try:
        ds = xr.open_dataset(file)
        z = ds['latent_variables'].values[1].reshape(100)
        ds.close()
    except FileNotFoundError:
        print(folder, " not found")
    return z

def slerp(val, low, high):
    """Spherical interpolation. val has a range of 0 to 1.
        From Tom White 2016
    """
    if val <= 0:
        return low
    elif val >= 1:
        return high
    elif np.allclose(low, high):
        return low
    omega = np.arccos(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega)/so * high

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
    
    if args.cluster:
        z1_file = os.path.expandvars("$PBS_O_WORKDIR/deepflow/runs/flowwells_adam_gauss_bce/run_9/iteration_243.nc")
        z2_file = os.path.expandvars("$PBS_O_WORKDIR/deepflow/runs/flowwells_adam_gauss_bce/run_14/iteration_61.nc")
        z3_file = os.path.expandvars("$PBS_O_WORKDIR/deepflow/runs/flowwells_adam_gauss_bce/run_87/iteration_269.nc")

        output_path_1 = os.path.expandvars("$PBS_O_WORKDIR/deepflow/runs/interpolations/interpolation_9_14")
        output_path_2 = os.path.expandvars("$PBS_O_WORKDIR/deepflow/runs/interpolations/interpolation_14_87")
        output_path_3 = os.path.expandvars("$PBS_O_WORKDIR/deepflow/runs/interpolations/interpolation_87_9")

    else:
        z1_file = "./results/runs/low_perm/flowwells_adam_bce/min_map/run_9/iteration_243.nc"
        z2_file = "./results/runs/low_perm/flowwells_adam_bce/min_map/run_14/iteration_61.nc"
        z3_file = "./results/runs/low_perm/flowwells_adam_bce/min_map/run_87/iteration_269.nc"
    
    z_files = [[z1_file, z2_file], [z2_file, z3_file], [z3_file, z1_file]]
    paths = [output_path_1, output_path_2, output_path_3]

    z1 = get_latent_vector(z_files[args.seed][0])
    z2 = get_latent_vector(z_files[args.seed][1])

    vals = np.linspace(0, 1, 101)

    z_int = [torch.from_numpy(slerp(val, z1, z2)).view(1, 50, 1, 2) for val in vals]

    states = interpolate(args, z_int, generator, paths[args.seed])

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
