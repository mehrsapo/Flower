# AFHQ-Cat tuned hyperparameters (highest PSNR on val)
dataset=afhq_cat   ## celeba/celebahq also supported, but params below are AFHQ-Cat-specific
model=ot           ## rectified for celebahq, gradient_step for method=pnp_gs (Hurault) or diffusion for method=pnp_diff (Zhu), ot otherwise.
eval_split=test
max_batch=100
batch_size_ip=1

# ### PNP FLOW  (alpha & steps from table; N -> steps_pnp)
method=pnp_flow
problem=denoising
problem=inpainting   # box inpainting
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lr_pnp 1.0 alpha 0.5  num_samples 1 max_batch ${max_batch} batch_size_ip ${batch_size_ip} steps_pnp 100 device cuda:1
problem=random_inpainting
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lr_pnp 1.0 alpha 0.01 num_samples 1 max_batch ${max_batch} batch_size_ip ${batch_size_ip} steps_pnp 200 device cuda:1

