# DeepFlow
### History Matching in the Space of Deep Generative Models

Authors: [Lukas Mosser](https://twitter.com/porestar), [Olivier Dubrule](https://www.imperial.ac.uk/people/o.dubrule), [Martin J. Blunt](https://www.imperial.ac.uk/people/m.blunt) 

[Pytorch](https://pytorch.org) implementation of [DeepFlow: History Matching in the Space of Deep Generative Models](https://arxiv.org/abs/1905.05749)

## Model Architecture
<img src="https://github.com/LukasMosser/DeepFlow/raw/master/results/figures/overview.png" width="800">

The model architecture consists of two parts: the generative adversarial network (implemented in Pytorch) and the forward solver ([MRST](https://www.sintef.no/projectweb/mrst/)).
The coupling between the two is implemented in ```deepflow.mrst_coupling.PytorchMRSTCoupler``` and defines a fully differentiable computational graph.

## Traversing the latent space while "History Matching"

<img src="https://github.com/LukasMosser/DeepFlow/raw/master/results/animations/evolution_facies.gif" width="400">

A visualization of the intermediate geological models obtained during the optimisation process.

## Interpolation between MAP solutions

<img src="https://github.com/LukasMosser/DeepFlow/raw/master/results/animations/interpolated_1_4_5_1.gif" width="400">

Interpolation in latent space between three MAP estimates shown in the publication (Figure 9a-b)

## Usage

To perform the inversion using the available pre-trained generator network use ``` run_deepflow.sh ```  
This requires a current version of Matlab or Octave available on the PATH.  
Interpolation is performed by running ```interpolation.py``` using the example bash file ```run_deepflow_interpolation.sh```.

## Trained Models
Pre-trained models are available in the  [checkpoints](checkpoints/) directory.

## Results and Data

A subset of the results is available in this [Google Drive](https://drive.google.com/drive/folders/1xLkLwDxAGVmfz-o2DzImgr8fP0fQNHW4?usp=sharing)  
The full dataset of the computations is multiple terrabyte in size and cannot be shared.  
Computing each run was made reproducible by setting the run-number = seed command-line argumen.  
Computations were performed on Imperial College CX1 supercomputing facilities.
Total duration: 3 days wall-time on 100 servers @ 4-cores each.  

## Matlab / Octave Compatibility
The reservoir simulator that solves the two-phase flow problem and provides gradients via the adjoint [MRST](https://www.sintef.no/projectweb/mrst/)
requires a Matlab license to run, but should be fully compatbile with [GNU Octave](https://www.gnu.org/software/octave/)

## Citing

```
@ARTICLE{2019arXiv190505749M,
       author = {{Mosser}, Lukas and {Dubrule}, Olivier and {Blunt}, Martin J.},
        title = "{DeepFlow: History Matching in the Space of Deep Generative Models}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Machine Learning, Computer Science - Computer Vision and Pattern Recognition, Physics - Computational Physics, Physics - Geophysics, Statistics - Machine Learning},
         year = "2019",
        month = "May",
          eid = {arXiv:1905.05749},
        pages = {arXiv:1905.05749},
        archivePrefix = {arXiv},
       eprint = {1905.05749},
        primaryClass = {cs.LG},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2019arXiv190505749M},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## Acknowledgements

The author would like to acknolwedge the developers of the [Matlab Reservoir Simulator Toolbox](https://www.sintef.no/projectweb/mrst/).  
If you use their software, please acknowledge them in your references.  
O. Dubrule would like to thank Total for seconding him as a visiting professor at Imperial College London.

## License

[MIT](LICENSE)


