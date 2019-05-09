import torch
import scipy.io as io
import numpy as np
import subprocess as proc
import matplotlib.pyplot as plt
import os 

def load_production_data(fname, name='ws_ref'):
    properties = ["bhp", "qOr", "qWr", "wcut"]
    wells = [0, 1]
    ws = io.loadmat(fname)

    well_props = []
    for well in wells:
        prop_temp = []
        for prop in properties:
            prop_series = np.array([t[0][0][well][prop] for t in ws[name]]).flatten()
            prop_temp.append(prop_series)
        well_props.append(prop_temp)

    return np.array(well_props)

def load_gradients(fname):
    print(fname)
    grad = io.loadmat(fname)
    grad_pore = grad['sens']['porevolume'][0, 0].reshape(64, 128)
    grad_permx = grad['sens']['permx'][0, 0].reshape(64, 128)
    grad_permy = grad['sens']['permy'][0, 0].reshape(64, 128)
    grad_permz = grad['sens']['permz'][0, 0].reshape(64, 128)
    return np.array([grad_pore, grad_permx, grad_permy, grad_permz])

class PytorchMRSTCoupler(torch.autograd.Function):
        @staticmethod
        def forward(ctx, k, poro, args):
            k_pth = k.detach().numpy()
            poro_pth = poro.detach().numpy()

            k_np = np.expand_dims(k_pth, 4).astype(np.float64)
            poro_np = np.expand_dims(poro_pth, 4).astype(np.float64)

            out = {'perm': k_np, 'poro': poro_np}
            io.savemat(os.path.join(args['matlab_path'], 'utils/synthetic/synthetic.mat'), {'rock': out})

            proc.call(args['command']+args['call'])
            
            sens = io.loadmat(os.path.join(args['matlab_path'], 'utils/synthetic/grad.mat'))
            poro_sens = sens['sens']['porevolume'][0, 0].reshape(64, 128)           
            perm_sens = sens['sens']['permx'][0, 0].reshape(64, 128)

            grad_perm_torch = torch.from_numpy(perm_sens.T).unsqueeze(0).unsqueeze(0)
            grad_poro_torch = torch.from_numpy(poro_sens.T).unsqueeze(0).unsqueeze(0)

            func = io.loadmat(os.path.join(args['matlab_path'], 'utils/synthetic/misfit.mat'))['misfitVal'][0, 0]

            ctx.save_for_backward(k, poro, grad_perm_torch, grad_poro_torch)
            
            return torch.from_numpy(np.array([func]))
        
        @staticmethod
        # This function has only a single output, so it gets only one gradient
        def backward(ctx, grad_output1):
            k, poro, grad_perm_torch, grad_poro_torch = ctx.saved_tensors
            
            grad_perm_torch /= grad_perm_torch.norm()
            grad_poro_torch /= grad_poro_torch.norm()

            return -grad_perm_torch.float(), -grad_poro_torch.float(), None #minus in front of grad poro!!!


if __name__=="__main__":
    poro = torch.ones(1, 1, 128, 64)*0.25
    perm = torch.ones(1, 1, 128, 64)*1e-13

    model = torch.cat([perm, poro], 1)
    model.requires_grad = True
    module = PytorchMRSTCoupler()
    
    layer =  module.apply
    func = layer(model)
    func.backward()
