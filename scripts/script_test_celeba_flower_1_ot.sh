# AFHQ-Cat tuned hyperparameters (highest PSNR on val)
dataset=celeba   ## celeba/celebahq also supported, but params below are AFHQ-Cat-specific
eval_split=test
max_batch=100
batch_size_ip=1

# ### PNP FLOW  (alpha & steps from table; N -> steps_pnp)
model=ot         
method=flower
# FLOWER 
problem=denoising
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} num_samples 1 max_batch ${max_batch} batch_size_ip ${batch_size_ip} steps 100 device cuda:0
problem=inpainting
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} num_samples 1 max_batch ${max_batch} batch_size_ip ${batch_size_ip} steps 100 device cuda:0
problem=gaussian_deblurring_FFT
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} num_samples 1 max_batch ${max_batch} batch_size_ip ${batch_size_ip} steps 100 device cuda:0
problem=superresolution
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} num_samples 1 max_batch ${max_batch} batch_size_ip ${batch_size_ip} steps 100 device cuda:0
problem=random_inpainting
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} num_samples 1 max_batch ${max_batch} batch_size_ip ${batch_size_ip} steps 100 device cuda:0
