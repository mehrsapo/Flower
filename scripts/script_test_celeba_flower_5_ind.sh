# AFHQ-Cat tuned hyperparameters (highest PSNR on val)
dataset=celeba   ## celeba/celebahq also supported, but params below are AFHQ-Cat-specific
eval_split=test
max_batch=100
batch_size_ip=1

# ### PNP FLOW  (alpha & steps from table; N -> steps_pnp)
model=flow_indp         
method=flower
# FLOWER 
problem=denoising
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} num_samples 5 max_batch ${max_batch} batch_size_ip ${batch_size_ip} steps 100 device cuda:3
problem=inpainting
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} num_samples 5 max_batch ${max_batch} batch_size_ip ${batch_size_ip} steps 100 device cuda:3
problem=gaussian_deblurring_FFT
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} num_samples 5 max_batch ${max_batch} batch_size_ip ${batch_size_ip} steps 100 device cuda:3
problem=superresolution
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} num_samples 5 max_batch ${max_batch} batch_size_ip ${batch_size_ip} steps 100 device cuda:3
problem=random_inpainting
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} num_samples 5 max_batch ${max_batch} batch_size_ip ${batch_size_ip} steps 100 device cuda:3
