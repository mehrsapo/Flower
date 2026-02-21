# AFHQ-Cat tuned hyperparameters (highest PSNR on val)
dataset=afhq_cat   ## celeba/celebahq also supported, but params below are AFHQ-Cat-specific
eval_split=test
max_batch=4
batch_size_ip=1

# ### PNP FLOW  (alpha & steps from table; N -> steps_pnp)
model=flow_indp         
method=flower_cov
# FLOWER 
problem=inpainting
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} num_samples 200 max_batch ${max_batch} batch_size_ip ${batch_size_ip} steps 100 device cuda:3
