import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score


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
    logProb = pdf.log_prob(z.view(1, -1)).mean(dim=1)
    prior_loss = -alpha*logProb.mean()
    return prior_loss