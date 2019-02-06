from torch.optim import Optimizer
from torch.distributions import Normal
import numpy as np
import torch

class MALA(Optimizer):
    r"""Implements the Metropolis Adjusted Langevin Algorithm
                  z = (1-lambda)*z - lr/2 * g + N(0, sqrt(lr))
        where z, g, and N denote the latent vector, gradient, and the Gaussian distribution. 
        lambda is the weight decay parameter and lr the step size.
        Usually combined with a step decline lr_t+1 = lr* t/(t+1) 
    """

    def __init__(self, params, lr=None, weight_decay=0):
        if lr is not None and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, weight_decay=weight_decay)
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
                    torch.ones(size) * np.sqrt(group['lr'])
                )
                p.data.add_(-group['lr']/2., d_p.data)
                p.data.add_(noise.sample())
        return loss