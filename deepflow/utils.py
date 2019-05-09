import numpy as np
import torch
import random
import xarray as xr

from deepflow.generator import PermeabilityGeneratorMRST as PermeabilityGenerator
from deepflow.networks import GeneratorMultiChannel


def print_header():
    print("")
    print("""\
       /$$$$$$$                                /$$$$$$$$ /$$                        
      | $$__  $$                              | $$_____/| $$                        
      | $$  \ $$  /$$$$$$   /$$$$$$   /$$$$$$ | $$      | $$  /$$$$$$  /$$  /$$  /$$
      | $$  | $$ /$$__  $$ /$$__  $$ /$$__  $$| $$$$$   | $$ /$$__  $$| $$ | $$ | $$
      | $$  | $$| $$$$$$$$| $$$$$$$$| $$  \ $$| $$__/   | $$| $$  \ $$| $$ | $$ | $$
      | $$  | $$| $$_____/| $$_____/| $$  | $$| $$      | $$| $$  | $$| $$ | $$ | $$
      | $$$$$$$/|  $$$$$$$|  $$$$$$$| $$$$$$$/| $$      | $$|  $$$$$$/|  $$$$$/$$$$/
      |_______/  \_______/ \_______/| $$____/ |__/      |__/ \______/  \_____/\___/ 
                                    | $$                                            
                                    | $$                                            
                                    |__/                                            
    """)
    print("")


def set_seed(seed):
    """
    Set the random number generator and turn off the cudnn benchmarks and backends to make truly deterministic
    For reproducibility purposes
    :param seed: random number generator seed
    :return: True on Success
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    return True


def load_generator(checkpoints_path):
    """
    Helper function to load the generator and set it to eval mode.
    Also initialises the PermeabilityGenerator module to go from earth model output to permeability/porosity values
    :param checkpoints_path: path to the generator checkpoint
    :return: initialised PermeabilityGenerator object
    """
    generator = GeneratorMultiChannel()
    new_state_dict = torch.load(checkpoints_path)
    generator.load_state_dict(new_state_dict)
    generator.cpu()
    generator.eval()

    gen = PermeabilityGenerator(generator)

    return gen


def report_latent_vector_stats(iteration, z, after):
    """
    Debugging helper function for outputing intermediate latent vector statistics.
    :param iteration: Current optimisation iteration.
    :param z: latent vector
    :param after: which phase of the optimization (well loss, flow loss, prior loss etc)
    :return: True on success
    """
    print("Latent Vector Stats after "+after+" Iteration: ", iteration,  " Min: %1.2f" % z.min().item(), 
    " Max: %1.2f" % z.max().item(), " Mean: %1.2f" % z.mean().item(), 
    " Std: %1.2f" % z.std().item(), " Norm: %2.2f" % z.norm().item(), " Current latent gradient norm: %2.2f" %z.grad.norm())
    return True


def get_latent_vector(file):
    """
    Loads a latent vector from a stored xarray.
    Used in interpolation of latent vectors
    :param file: xarray file path
    :return: latent vector
    """
    z = None
    try:
        ds = xr.open_dataset(file)
        z = ds['latent_variables'].values[1].reshape(100)
        ds.close()
    except FileNotFoundError:
        print(folder, " not found")
    return z


def slerp(val, low, high):
    """

    Spherical interpolation. val has a range of 0 to 1.
    From Tom White 2016

    :param val: interpolation mixture value
    :param low: first latent vector
    :param high: second latent vector
    :return: 
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
