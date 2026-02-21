# AFHQ-Cat tuned hyperparameters (highest PSNR on val)
dataset=afhq_cat   ## celeba/celebahq also supported, but params below are AFHQ-Cat-specific
model=ot           ## rectified for celebahq, gradient_step for method=pnp_gs (Hurault) or diffusion for method=pnp_diff (Zhu), ot otherwise.
eval_split=test
max_batch=100
batch_size_ip=1

### FLOW PRIORS  (matches table)
method=flow_priors
problem=denoising
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} eta 0.01 lmbda 100    max_batch ${max_batch} batch_size_ip ${batch_size_ip} 
problem=gaussian_deblurring_FFT
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} eta 0.01 lmbda 1000   max_batch ${max_batch} batch_size_ip ${batch_size_ip}
problem=superresolution
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} eta 0.1  lmbda 10000 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
problem=inpainting   # box inpainting
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} eta 0.01 lmbda 10000 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
problem=random_inpainting
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} eta 0.01 lmbda 10000 max_batch ${max_batch} batch_size_ip ${batch_size_ip}


## D Flow  (lambda/alpha/n_iter from table)
method=d_flow
problem=denoising
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lmbda 0.001 alpha 0.1 max_batch ${max_batch} batch_size_ip ${batch_size_ip} max_iter 3
problem=gaussian_deblurring_FFT
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lmbda 0.01  alpha 0.5 max_batch ${max_batch} batch_size_ip ${batch_size_ip} max_iter 20
problem=superresolution
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lmbda 0.001 alpha 0.1 max_batch ${max_batch} batch_size_ip ${batch_size_ip} max_iter 20
problem=inpainting   # box inpainting
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lmbda 0.01  alpha 0.1 max_batch ${max_batch} batch_size_ip ${batch_size_ip} max_iter 9
problem=random_inpainting
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lmbda 0.001 alpha 0.1 max_batch ${max_batch} batch_size_ip ${batch_size_ip} max_iter 20
