# AFHQ-Cat tuned hyperparameters (highest PSNR on val)
dataset=afhq_cat   ## celeba/celebahq also supported, but params below are AFHQ-Cat-specific
model=flow_indp           ## rectified for celebahq, gradient_step for method=pnp_gs (Hurault) or diffusion for method=pnp_diff (Zhu), ot otherwise.
eval_split=test
max_batch=1
batch_size_ip=1

# ### PNP FLOW  (alpha & steps from table; N -> steps_pnp)
# FLOWER 
method=flower_indp
problem=denoising
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} num_samples 1 max_batch ${max_batch} batch_size_ip ${batch_size_ip} steps 100
