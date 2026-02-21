# Flower

This GitHub repository contains the code for our ICLR 2026 Flower [paper](https://arxiv.org/abs/2509.26287), a method that aims to solve inverse problems with pretrained flow matching models through a Bayesian viewpoint.

Flower consists of three main steps. 
- Destination Estimation 
- Destination Refinement
- Time Progression 


<img src="flower_demo/flower_steps.png" scale=0.8/>
<img src="flower_demo/batch001_grid.png" scale=0.6/>

## 1. Getting started
To get started, clone and install the repository with:

```
cd Flower
pip install -e .
```

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

## Acknowledgements
This repository builds upon the following publicly available codes:
- [PnP-Flow](https://arxiv.org/abs/2410.02423) available at https://github.com/annegnx/PnP-Flow,
which builds upon:
- [PnP-GS](https://openreview.net/pdf?id=fPhKeld3Okz) available at https://github.com/samuro95/GSPnP
- [DiffPIR](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/papers/Zhu_Denoising_Diffusion_Models_for_Plug-and-Play_Image_Restoration_CVPRW_2023_paper.pdf) from the [DeepInv](https://deepinv.github.io/deepinv/stubs/deepinv.sampling.DiffPIR.html#deepinv.sampling.DiffPIR) library
- The folder ImageGeneration is copied from the [Rectified Flow](https://github.com/gnobitab/RectifiedFlow) repository.
- We thank Anne Gagneux and Ségolène Martin for their assistance in reproducing PnP-Flow results.

