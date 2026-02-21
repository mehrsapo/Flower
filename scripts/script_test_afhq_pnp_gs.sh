### PNP GRADIENT STEP (Hurault)  (gamma/alpha/sigma_f/n_iter from table)
dataset=afhq_cat   ## celeba/celebahq also supported, but params below are AFHQ-Cat-specific
model=ot           ## rectified for celebahq, gradient_step for method=pnp_gs (Hurault) or diffusion for method=pnp_diff (Zhu), ot otherwise.
eval_split=test
max_batch=100
batch_size_ip=1

method=pnp_gs
model=gradient_step
problem=denoising
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lr_pnp 1.0 alpha 1.0 max_batch ${max_batch} batch_size_ip ${batch_size_ip} algo pgd max_iter 1  sigma_factor 1.0
problem=gaussian_deblurring_FFT
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lr_pnp 2.0 alpha 0.3 max_batch ${max_batch} batch_size_ip ${batch_size_ip} algo pgd max_iter 60 sigma_factor 1.8
problem=superresolution
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lr_pnp 2.0 alpha 1.0 max_batch ${max_batch} batch_size_ip ${batch_size_ip} algo pgd max_iter 50 sigma_factor 5.0
problem=random_inpainting
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} max_batch ${max_batch} batch_size_ip ${batch_size_ip} algo hqs max_iter 23
