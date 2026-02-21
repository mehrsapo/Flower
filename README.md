# Flower

This GitHub repository contains the code for our ICLR 2026 Flower [paper](https://arxiv.org/abs/2509.26287), a method that aims to solve inverse problems with pretrained flow matching models through a Bayesian viewpoint.

Flower consists of three main steps. 
- Destination Estimation 
- Destination Refinement
- Time Progression 


<img src="flower_demo/flower_steps.png" scale=0.8/>
<img src="flower_demo/batch001_grid.png" scale=0.6/>

## 1. Getting started
To get started, clone and install the repository with 

```
cd Flower
pip install -e .
```
### 1.1. Requirements

- torch 1.13.1 (or later)
- torchvision
- tqdm
- numpy
- pandas
- pyyaml
- scipy
- torchdiffeq
- deepinv

### 1.2. Download datasets and pretrained models

We provide a script to download datasets used in PnP-Flow and the corresponding pre-trained networks. The datasets and network checkpoints will be downloaded and stored in the `data` and `model` directories, respectively.

<b>CelebA.</b> To download the [CelebA](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) dataset and the pre-trained OT FM network (U-Net), run the following commands:
```bash
bash download.sh celeba-dataset
bash download.sh pretrained-network-celeba
```

<b>AFHQ-CAT.</b> To download the [AFHQ-CAT](https://github.com/clovaai/stargan-v2/blob/master/README.md#animal-faces-hq-dataset-afhq) dataset and the pre-trained OT FM network (U-Net), run the following commands:
```bash
bash download.sh afhq-cat-dataset
bash download.sh pretrained-network-afhq-cat
```

Note that as the dataset AFHQ-Cat doesn't have a validation split, we create one when downloading the dataset. 

Alternatively, the FM models can directly be downloaded here: [CelebA model](https://drive.google.com/file/d/1ZZ6S-PGRx-tOPkr4Gt3A6RN-PChabnD6/view?usp=drive_link), [AFHQ-Cat model](https://drive.google.com/file/d/1FpD3cYpgtM8-KJ3Qk48fcjtr1Ne_IMOF/view?usp=drive_link), [MNIST-Dirichlet model](https://drive.google.com/file/d/1If5gkWEfChJHc8v8CCEhGhEeeAqsxKTz/view?usp=drive_link)

And the denoisers for the PnP-GS method here: [CelebA model](https://drive.google.com/file/d/1ZqBeafErEogaXFupW0ZSLL7P9QoRA-lN/view?usp=drive_link), [AFHQ-Cat model](https://drive.google.com/file/d/17AXI9p17c7h_xaI19qDcTT2u9_wu0DQY/view?usp=drive_link)

## 2. Training

You can also use the code to train your own OT Flow Matching model.

You can modify the config options directly in the main_config.yaml file located in ```config/```. Alternatively, config keys can be given as options directly in the command line.

For example, to train the generative flow matching model (here, the U-net is the velocity) on CelebA, with a Gaussian latent distribution, run:
```python
python main.py --opts dataset celeba train True eval False batch_size 128 num_epoch 100
```
At each 5 epochs, the model is saved in ```./model/celeba/gaussian/ot```. Generated samples are saved in ```./results/celeba/gaussian/ot```.

### Computing generative model scores

After the training, the final model is loaded and can be used for generating samples / solving inverse problems. You can compute the full FID (based on 50000 generated samples), the Vendi score, and the Slice Wasserstein score running
```python
python main.py --opts dataset mnist train False eval True compute_metrics True solve_inverse_problem False
```
## 3. Solving inverse problems

The available inverse problems are:
- Denoising --> set ```problem: 'denoising'```
- Gaussian deblurring --> set ```problem: 'gaussian_deblurring'```
- Super-resolution --> set ```problem: 'superresolution'```
- Box inpainting --> set ```problem: 'inpainting'```
- Random inpainting --> set ```problem: 'random_inpainting'```
- Free-form inpainting --> set ```problem: 'paintbrush_inpainting'```

The parameters of the inverse problems (e.g., noise level) can be adjusted manually in the ```main.py``` file.

The available methods are
- ```flower``` (our method)
- ```pnp_flow``` (from this [paper](https://arxiv.org/pdf/2402.14017))
- ```ot_ode``` (from this [paper](https://openreview.net/forum?id=PLIt3a4yTm&referrer=%5Bthe%20profile%20of%20Ashwini%20Pokle%5D(%2Fprofile%3Fid%3D~Ashwini_Pokle1)))
- ```d_flow``` (from this [paper](https://arxiv.org/pdf/2402.14017))
- ```flow_priors``` (from this [paper](https://arxiv.org/abs/2405.18816))
- ```pnp_diff``` (from this [paper](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/papers/Zhu_Denoising_Diffusion_Models_for_Plug-and-Play_Image_Restoration_CVPRW_2023_paper.pdf))
- ```pnp_gs``` (from this [paper](https://openreview.net/pdf?id=fPhKeld3Okz))


### 3.2. Evaluation on the test set

Use the bash files ```scripts/``.

Visual results will be saved in ```results/``` folder. 

## Acknowledgements
This repository builds upon the following publicly available codes:
- [PnP-Flow](https://arxiv.org/abs/2410.02423) available at https://github.com/annegnx/PnP-Flow. 
which builds upons: 
- [PnP-GS](https://openreview.net/pdf?id=fPhKeld3Okz) available at https://github.com/samuro95/GSPnP
- [DiffPIR](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/papers/Zhu_Denoising_Diffusion_Models_for_Plug-and-Play_Image_Restoration_CVPRW_2023_paper.pdf) from the [DeepInv](https://deepinv.github.io/deepinv/stubs/deepinv.sampling.DiffPIR.html#deepinv.sampling.DiffPIR) library
- The folder ImageGeneration is copied from [Rectified Flow](https://github.com/gnobitab/RectifiedFlow) repository.
- We thank Anne Gagneux and S´egol`ene Martin for their assistance in reproducing pnp-flow results. 

