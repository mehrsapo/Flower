# AFHQ-Cat tuned hyperparameters (highest PSNR on val)
dataset=afhq_cat   ## celeba/celebahq also supported, but params below are AFHQ-Cat-specific
model=ot           ## rectified for celebahq, gradient_step for method=pnp_gs (Hurault) or diffusion for method=pnp_diff (Zhu), ot otherwise.
eval_split=test
max_batch=100
batch_size_ip=1

# ### OT ODE  (t0=start_time, gamma: sqrt(t)->gamma_t, constant->constant)
method=ot_ode
problem=denoising
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} start_time 0.3 max_batch ${max_batch} batch_size_ip ${batch_size_ip} gamma gamma_t device cuda:1
problem=gaussian_deblurring_FFT
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} start_time 0.3 max_batch ${max_batch} batch_size_ip ${batch_size_ip} gamma gamma_t device cuda:1
problem=superresolution
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} start_time 0.1 max_batch ${max_batch} batch_size_ip ${batch_size_ip} gamma constant device cuda:1
problem=inpainting   # box inpainting
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} start_time 0.1 max_batch ${max_batch} batch_size_ip ${batch_size_ip} gamma gamma_t device cuda:1
problem=random_inpainting
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} start_time 0.1 max_batch ${max_batch} batch_size_ip ${batch_size_ip} gamma constant device cuda:1
