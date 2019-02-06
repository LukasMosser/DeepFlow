import numpy as np
import torch
import random

from deepflow.generator import PermeabilityGeneratorMRST as PermeabilityGenerator
from deepflow.networks import GeneratorMultiChannel

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled   = False

    return True


def load_generator(checkpoints_path):
    generator = GeneratorMultiChannel()
    new_state_dict = torch.load(checkpoints_path)
    generator.load_state_dict(new_state_dict)
    generator.cpu()
    generator.eval()

    gen = PermeabilityGenerator(generator)

    return gen

def report_latent_vector_stats(iteration, z, after):
    print("Latent Vector Stats after "+after+" Iteration: ", iteration,  " Min: %1.2f" % z.min().item(), 
    " Max: %1.2f" % z.max().item(), " Mean: %1.2f" % z.mean().item(), 
    " Std: %1.2f" % z.std().item(), " Norm: %2.2f" % z.norm().item(), " Current latent gradient norm: %2.2f" %z.grad.norm())
    return True
