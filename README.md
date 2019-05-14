# DeepFlow
## History Matching in the Space of Deep Generative Models

Authors: [Lukas Mosser](https://twitter.com/porestar), [Olivier Dubrule](emailto:o.dubrule@imperial.ac.uk), [Martin J. Blunt]((emailto:m.blunt@imperial.ac.uk))  

[Pytorch](https://pytorch.org) implementation of [DeepFlow: History Matching in the Space of Deep Generative Models]()

### Model Architecture
<img src="https://github.com/LukasMosser/DeepFlow/results/figures/overview.png" width="400">

The model architecture consists of two parts: the generative adversarial network (implemented in Pytorch) and the forward solver ([MRST](https://www.sintef.no/projectweb/mrst/)).
The coupling between the two is implemented in ```deepflow.mrst_coupling.PytorchMRSTCoupler``` and defines a fully differentiable computational graph.

### Example Inversion

<img src="https://github.com/LukasMosser/DeepFlow/results/figures/evolution_facies.gif" width="300">

A visualization of the intermediate geological models obtained during the optimisation process.

### Usage

To perform the inversion using the available pre-trained generator network use ``` run_deepflow.sh ```  
This requires a current version of Matlab or Octave available on the PATH.

### Trained Models
Pre-trained models are available in the  [checkpoints](checkpoints/) directory.

### Matlab / Octave Compatibility
The reservoir simulator that solves the two-phase flow problem and provides gradients via the adjoint [MRST](https://www.sintef.no/projectweb/mrst/)
requires a Matlab license to run, but should be fully compatbile with [GNU Octave](https://www.gnu.org/software/octave/)
### Acknowledgments

The author would like to acknolwedge the developers of the [Matlab Reservoir Simulator Toolbox](https://www.sintef.no/projectweb/mrst/).
If you use their software, please acknowledge them in your references.

### License

[MIT](LICENSE)


