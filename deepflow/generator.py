import torch
import torch.nn as nn


class PermeabilityGeneratorMRST(nn.Module):
    def __init__(self, generator):
        """
        Initializes the permeability porosity transform module
        :param generator: pre-trained GAN generator
        """
        super(PermeabilityGeneratorMRST, self).__init__()
        self.generator = generator

    def forward(self, z, a=0.001, b=1e-12, c=0.3, d=0.1): #high-perm-case: a=0.1, b=1e-13, c=0.3, d=0.1
        """
        Transform the incoming earth model from the GAN generator to permeability and porosity values.
        :param z: latent vector
        :param a: Permeability Coefficient a
        :param b: Permeability Coefficient b
        :param c: Porosity Coefficient c
        :param d: Porosity Coefficient d
        :return: permeability, porosity and facies probability
        """
        x = self.generator(z)
        x_k = (x[:, 0, :, :].transpose(2, 1)/2.+0.5)
        k = (x_k+a)*b

        x_poro = (x[:, 1, :, :].transpose(2, 1)/2.+0.5)
        poro = (x_poro*c)+d
        return k.unsqueeze(0), poro.unsqueeze(0), x_k

