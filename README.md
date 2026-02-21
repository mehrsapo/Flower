# FLOWER: A Flow Matching Solver for Inverse Problems
[![arXiv](https://img.shields.io/badge/arXiv-2509.26287-b31b1b.svg)](https://arxiv.org/abs/2509.26287)
[![OpenReview](https://img.shields.io/badge/OpenReview-ICLR_2026-blue.svg)](https://openreview.net/forum?id=QGd34p02mI)
[![Project Page](https://img.shields.io/badge/Project%20Page-Online-green.svg)](https://mehrsapo.github.io/Flower/)

This GitHub repository contains the code for our ICLR 2026 Flower [paper](https://openreview.net/forum?id=QGd34p02mI), a method that aims to solve inverse problems with pretrained flow matching models through a Bayesian viewpoint.

Flower consists of three main steps. 
- Destination Estimation 
- Destination Refinement
- Time Progression 

which jointly jointly approximate posterior sampling and solve linear inverse problems.

<img src="flower_demo/flower_steps.png" scale=0.8/>

A visual example for the solution path of Flower for box inpainting:

<img src="flower_demo/batch001_grid.png" scale=0.6/>

## 1. Getting started
To get started, clone the repository and create the conda environment:

```bash
cd Flower
conda env create -f environment.yml
conda activate flower
```

This will install all dependencies and the package in editable mode. The environment uses Python 3.12.2 with PyTorch installed via conda (CUDA 12.4 by default — edit `environment.yml` to match your CUDA version, or replace `pytorch-cuda=12.4` with `cpuonly` for a CPU-only setup).

### 1.1. Download data

To download the [CelebA](https://www.kaggle.com/datas/jessicali9530/celeba-data) and [AFHQ-Cat](https://github.com/clovaai/stargan-v2/blob/master/README.md#animal-faces-hq-data-afhq) datas, run the command:
```bash
bash download_data.sh
```
Note that since the AFHQ-Cat data does not have a test split, we create one when downloading the data.

### 1.2. Download pretrained models

To download all the pretrained model weights, run the command:

```bash
bash download_models.sh
```

## 2. Demo notebooks

Two interactive Jupyter notebooks are included under the
`flower_demo/` directory to help you get a feel for how the Flower
algorithm is used in practice. 

### 2.1 CS‑MRI with radial sampling

`flower_demo/exps_cs_mri_radial.ipynb` starts by loading an
`afhq_cat`‑trained Flower model (the default OT variant) and a single
RGB test image.  A radial undersampling mask is read from
`radial_mask.mat`, and forward/adjoint MRI operators are defined using
PyTorch FFTs.  The notebook implements the conjugate‑gradient solver
used by Flower and then runs two reconstruction modes:
`flower` (isotropic covariance approximation $\gamma = 0$) and `flower_cov` (full
posterior covariance sampling $\gamma=1$).  After the iterative posterior sampler
runs for 100 steps, the ground truth, zero‑filled adjoint image, and
both Flower reconstructions are displayed side‑by‑side, along with the
sampling sparsity and noise level.

### 2.2 Non‑isotropic denoising

`flower_demo/exps_non_iso_noise.ipynb` demonstrates the use of Flower on two CelebA images affected by spatially
varying noise.  The notebook constructs a per‑pixel variance map where
the centre region has four times the noise standard deviation of the
border, and it extends the conjugate‑gradient solver to handle this
non‑isotropic covariance directly.  

*NOTE:* any new inverse problem can be tackled simply by defining a
different linear forward operator `H` and its adjoint `Hᵗ`—the CS‑MRI
notebook shows a masked FFT example, but you could equally replace it
with convolution, subsampling, or any other linear map and then run
the same code.

## 3. Reproduction of paper results for solving inverse problems
Our comparisons build upon the [PnP-Flow]( https://github.com/annegnx/PnP-Flow) benchmark. 

Use the bash scripts in the ```scripts/``` folder.

Visual and numerical results will be saved in the ```results/``` folder.

The available methods are
  - ```flower``` (our method default with $\gamma = 0$)
  - ```flower_cov``` (our method with $\gamma = 1$)
  - ```pnp_flow``` (from this [paper](https://arxiv.org/pdf/2402.14017))
  - ```ot_ode``` (from this [paper](https://openreview.net/forum?id=PLIt3a4yTm&referrer=%5Bthe%20profile%20of%20Ashwini%20Pokle%5D(%2Fprofile%3Fid%3D~Ashwini_Pokle1)))
  - ```d_flow``` (from this [paper](https://arxiv.org/pdf/2402.14017))
  - ```flow_priors``` (from this [paper](https://arxiv.org/abs/2405.18816))
  - ```pnp_diff``` (from this [paper](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/papers/Zhu_Denoising_Diffusion_Models_for_Plug-and-Play_Image_Restoration_CVPRW_2023_paper.pdf))
  - ```pnp_gs``` (from this [paper](https://openreview.net/pdf?id=fPhKeld3Okz))

Note that ```flower``` and ```flower_cov``` support two modes for the pretrained flow model: 1. ```ot```: with optimal transport coupling (used for comparisons), and 2. ```flow_indp```: with no optimal transport coupling during training. Read the paper for more details.

The available inverse problems are:
  - Denoising -->  ```problem: 'denoising'```
  - Gaussian deblurring -->  ```problem: 'gaussian_deblurring'```
  - Super-resolution -->  ```problem: 'superresolution'```
  - Box inpainting -->  ```problem: 'inpainting'```
  - Random inpainting -->  ```problem: 'random_inpainting'```


## 4. Questions & Contact

  If you run into issues, have questions about the code, or would like to
  provide feedback, please don't hesitate to get in touch:

  📧 **Email:** mehrsapo@gmail.com

## 5. Citation

  If you use this code in your research, please consider citing our paper:

  ```
  @inproceedings{pourya2026flower,
    title={{FLOWER}: A Flow-Matching Solver for Inverse Problems},
    author={Mehrsa Pourya and Bassam El Rawas and Michael Unser},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=QGd34p02mI}
  }
  ```

  We'll be happy to help or point you in the right direction.

## Acknowledgements
This repository builds upon the following publicly available codes:
- [PnP-Flow](https://arxiv.org/abs/2410.02423) available at https://github.com/annegnx/PnP-Flow,

which builds upon:

- [PnP-GS](https://openreview.net/pdf?id=fPhKeld3Okz) available at https://github.com/samuro95/GSPnP
- [DiffPIR](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/papers/Zhu_Denoising_Diffusion_Models_for_Plug-and-Play_Image_Restoration_CVPRW_2023_paper.pdf) from the [DeepInv](https://deepinv.github.io/deepinv/stubs/deepinv.sampling.DiffPIR.html#deepinv.sampling.DiffPIR) library
- The folder ImageGeneration is copied from the [Rectified Flow](https://github.com/gnobitab/RectifiedFlow) repository.
- We thank Anne Gagneux and Ségolène Martin for their assistance in reproducing PnP-Flow results.

