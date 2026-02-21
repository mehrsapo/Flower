# Flower

This GitHub repository contains the code for our ICLR 2026 Flower [paper](https://arxiv.org/abs/2509.26287), a method that aims to solve inverse problems with pretrained flow matching models through a Bayesian viewpoint.

Flower consists of three main steps. 
- Destination Estimation 
- Destination Refinement
- Time Progression 


<img src="flower_demo/flower_steps.png" scale=0.8/>
<img src="flower_demo/batch001_grid.png" scale=0.6/>

## 1. Getting started
To get started, clone the repository and create the conda environment:

```bash
cd Flower
conda env create -f environment.yml
conda activate flower
```

This will install all dependencies and the package in editable mode. The environment uses Python 3.12.2 with PyTorch installed via conda (CUDA 12.4 by default — edit `environment.yml` to match your CUDA version, or replace `pytorch-cuda=12.4` with `cpuonly` for a CPU-only setup).

### 1.1. Download datasets

To download the [CelebA](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) and [AFHQ-Cat](https://github.com/clovaai/stargan-v2/blob/master/README.md#animal-faces-hq-dataset-afhq) datasets, run the command:
```bash
bash download_data.sh
```
Note that since the AFHQ-Cat dataset does not have a test split, we create one when downloading the dataset.

### 1.2. Download pretrained models

To download all the pretrained model weights, run the command:

```bash
bash download_models.sh
```

## 2. Reproduction of paper results for solving inverse problems
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
	- Denoising --> set ```problem: 'denoising'```
	- Gaussian deblurring --> set ```problem: 'gaussian_deblurring'```
	- Super-resolution --> set ```problem: 'superresolution'```
	- Box inpainting --> set ```problem: 'inpainting'```
	- Random inpainting --> set ```problem: 'random_inpainting'```
## 3. Demo notebooks

Two interactive Jupyter notebooks are included under the
`flower_demo/` directory to help you get a feel for how the FLOWER
algorithm is used in practice.  They are standalone scripts that
initialise the same configuration and model code used by the
reproduction scripts, then run small toy examples and plot the results.

### 3.1 CS‑MRI with radial sampling

`flower_demo/exps_cs_mri_radial.ipynb` starts by loading an
`afhq_cat`‑trained FLOWER model (the default OT variant) and a single
RGB test image.  A radial undersampling mask is read from
`radial_mask.mat`, and forward/adjoint MRI operators are defined using
PyTorch FFTs.  The notebook implements the conjugate‑gradient solver
used by FLOWER and then runs two reconstruction modes:
`flower` (isotropic covariance approximation) and `flower_cov` (full
posterior covariance sampling).  After the iterative posterior sampler
runs for 100 steps, the ground truth, zero‑filled adjoint image, and
both FLOWER reconstructions are displayed side‑by‑side, along with the
sampling sparsity and noise level.

### 3.2 Non‑isotropic denoising

`flower_demo/exps_non_iso_noise.ipynb` demonstrates the same flow
model on a pair of pre‑cropped CelebA images affected by spatially
varying noise.  The notebook constructs a per‑pixel variance map where
the centre region has four times the noise standard deviation of the
border, and it extends the conjugate‑gradient solver to handle this
non‑isotropic covariance directly.  A small 20‑step reverse‑diffusion
loop applies the FLOWER sampler, and the resulting noisy input and
reconstruction are plotted with PSNR values.

### Running the demos

To explore the notebooks yourself, launch a Jupyter server from the
repository root and open the `.ipynb` files::

```bash
cd Flower
conda activate flower           # or your preferred env
jupyter lab                    # or jupyter notebook
```

The cells are meant to be executed sequentially; the first cell adds
the project root to `sys.path` so that the package imports work
regardless of the working directory.  You can also run the notebooks
from the command line with `nbconvert` if you prefer to generate
static HTML outputs.

## Acknowledgements
This repository builds upon the following publicly available codes:
- [PnP-Flow](https://arxiv.org/abs/2410.02423) available at https://github.com/annegnx/PnP-Flow,
which builds upon:
- [PnP-GS](https://openreview.net/pdf?id=fPhKeld3Okz) available at https://github.com/samuro95/GSPnP
- [DiffPIR](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/papers/Zhu_Denoising_Diffusion_Models_for_Plug-and-Play_Image_Restoration_CVPRW_2023_paper.pdf) from the [DeepInv](https://deepinv.github.io/deepinv/stubs/deepinv.sampling.DiffPIR.html#deepinv.sampling.DiffPIR) library
- The folder ImageGeneration is copied from the [Rectified Flow](https://github.com/gnobitab/RectifiedFlow) repository.
- We thank Anne Gagneux and Ségolène Martin for their assistance in reproducing PnP-Flow results.

