dataset=celeba ## or celebahq or afhq_cat
model=ot  ## rectified for celebahq, gradient_step for method=pnp_gs (Hurault) or diffusion for method=pnp_diff (Zhu), ot otherwise.
eval_split=test
max_batch=100
batch_size_ip=1

# FLOWER 
# ### PNP FLOW
method=pnp_flow
problem=denoising
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lr_pnp 1.0 alpha 0.8 num_samples 1 max_batch ${max_batch} batch_size_ip ${batch_size_ip} steps_pnp 100 device 'cuda:3'
problem=gaussian_deblurring_FFT
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lr_pnp 1.0 alpha 0.01 num_samples 1 max_batch ${max_batch} batch_size_ip ${batch_size_ip} steps_pnp 100 device 'cuda:3'
problem=superresolution
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lr_pnp 1.0 alpha 0.3 num_samples 1 max_batch ${max_batch} batch_size_ip ${batch_size_ip} steps_pnp 100 device 'cuda:3'
problem=inpainting
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lr_pnp 1.0 alpha 0.5 num_samples 1 max_batch ${max_batch} batch_size_ip ${batch_size_ip} steps_pnp 100 device 'cuda:3'
problem=random_inpainting
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lr_pnp 1.0 alpha 0.01 num_samples 1 max_batch ${max_batch} batch_size_ip ${batch_size_ip} steps_pnp 100 device 'cuda:3'
