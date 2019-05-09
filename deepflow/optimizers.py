from torch.optim import Optimizer
from torch.distributions import Normal
import numpy as np
import torch
import math

class MALA(Optimizer):
    r"""Implements the Metropolis Adjusted Langevin Algorithm
                  z = (1-lambda)*z - lr/2 * g + N(0, sqrt(lr))
        where z, g, and N denote the latent vector, gradient, and the Gaussian distribution. 
        lambda is the weight decay parameter and lr the step size.
        Usually combined with a step decline lr_t+1 = lr* t/(t+1) 
    """

    def __init__(self, params, lr=None, weight_decay=0, eps3=0.0):
        if lr is not None and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, weight_decay=weight_decay, eps3=eps3)
        super(MALA, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(MALA, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                    
                size = d_p.size()
                noise = Normal(
                    torch.zeros(size),
                    torch.ones(size) * group['eps3']
                )
                p.data.add_(-group['lr'], d_p.data)
                p.data.add_(noise.sample())
        return loss

from torch.optim.optimizer import Optimizer, required

class pSGLD(Optimizer):
    def __init__(self, params, lr=required, alpha=0.99, eps=1e-5, centered=False, addnoise=True):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, centered=centered, addnoise=addnoise)
        super(pSGLD, self).__init__(params, defaults)

    def step(self, lr=None):
        for group in self.param_groups:
            if lr:
                group['lr'] = lr
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p.data)

                square_avg = state['square_avg']
                alpha = group['alpha']
                state['step'] += 1

                # sqavg x alpha + (1-alph) sqavg *(elemwise) sqavg
                square_avg.mul_(alpha).addcmul_(1 - alpha, d_p, d_p)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(1 - alpha, d_p)
                    avg = square_avg.cmul(-1, grad_avg, grad_avg).sqrt().add_(group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])

                if group['addnoise']:
                    size = d_p.size()
                    langevin_noise = Normal(
                        torch.zeros(size),
                        torch.ones(size).mul_(group['lr']).div_(avg).sqrt()
                    )
                    p.data.add_(-group['lr'], d_p.div_(avg) + langevin_noise.sample())
                else:
                    p.data.addcdiv_(-group['lr'], d_p, avg)

        return None

class pSGLDmod(Optimizer):
    """
    Barely modified version of pytorch SGD to implement pSGLD
    The RMSprop preconditioning code is mostly from pytorch rmsprop implementation.
    """

    def __init__(self, params, lr=required, alpha=0.99, eps=1e-8, centered=False, addnoise=True):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, centered=centered, addnoise=addnoise)
        super(pSGLDmod, self).__init__(params, defaults)
        
    def __setstate__(self, state):
        super(pSGLDmod, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('centered', False)

    def step(self, lr=None, add_noise = False):
        """
        Performs a single optimization step.
        """
        loss = None

        for group in self.param_groups:
            if lr:
                group['lr'] = lr
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p.data)
                        
                square_avg = state['square_avg']
                alpha = group['alpha']
                state['step'] += 1
                
                # sqavg x alpha + (1-alph) sqavg *(elemwise) sqavg
                square_avg.mul_(alpha).addcmul_(1-alpha, d_p, d_p)
                
                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(1-alpha, d_p)
                    avg = square_avg.cmul(-1, grad_avg, grad_avg).sqrt().add_(group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])
                    
                
                if group['addnoise']:
                    size = d_p.size()
                    langevin_noise = Normal(
                        torch.zeros(size),
                        torch.ones(size).div_(group['lr']).div_(avg).sqrt()
                    )
                    p.data.add_(-group['lr'],
                                d_p.div_(avg) + langevin_noise.sample())
                else:
                    #p.data.add_(-group['lr'], d_p.div_(avg))
                    p.data.addcdiv_(-group['lr'], d_p, avg)

        return loss