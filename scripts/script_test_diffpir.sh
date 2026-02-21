dataset=celeba ## or celebahq or afhq_cat
eval_split=test
max_batch=100
batch_size_ip=1


# ### PNP DIFFUSION
method=pnp_diff
model=diffusion
problem=denoising
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lmbda 1.0 zeta 1.0 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
problem=gaussian_deblurring_FFT
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lmbda 1000.0 zeta 1.0 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
problem=superresolution
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lmbda 100.0 zeta 1.0 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
problem=random_inpainting
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lmbda 1.0 zeta 1.0 max_batch ${max_batch} batch_size_ip ${batch_size_ip}


# AFHQ-Cat tuned hyperparameters (highest PSNR on val)
dataset=afhq_cat   ## celeba/celebahq also supported, but params below are AFHQ-Cat-specific
eval_split=test
max_batch=100
batch_size_ip=1

# ### PNP DIFFUSION (Zhu)  (zeta/lambda from table)
method=pnp_diff
model=diffusion
problem=denoising
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lmbda 1.0    zeta 1.0 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
problem=gaussian_deblurring_FFT
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lmbda 1000.0 zeta 1.0 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
problem=superresolution
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lmbda 100.0  zeta 1.0 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
problem=random_inpainting
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lmbda 1.0    zeta 1.0 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
