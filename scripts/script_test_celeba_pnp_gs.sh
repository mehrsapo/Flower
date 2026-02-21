dataset=celeba ## or celebahq or afhq_cat
eval_split=test
max_batch=100
batch_size_ip=1

### PNP GRADIENT STEP
method=pnp_gs
model=gradient_step
problem=gaussian_deblurring_FFT
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lr_pnp 2.0 alpha 0.5 max_batch ${max_batch} batch_size_ip ${batch_size_ip} algo pgd max_iter 35 sigma_factor 1.8 # alpha was mismatched with the paper it was 0.5, changed to 0.3
problem=superresolution
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lr_pnp 2.0 alpha 1.0 max_batch ${max_batch} batch_size_ip ${batch_size_ip} algo pgd max_iter 20 sigma_factor 3.0  # 1.8, table 3, i changed it
