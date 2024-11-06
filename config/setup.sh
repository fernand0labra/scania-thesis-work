#!/bin/bash

# ln -sf [OPTIONS] FILE LINK
# sudo ln -sf /bin/python3.8 /bin/python3
# sudo ln -sf /bin/python3.9 /bin/python3

# Install dependencies
pip install -r requirements.txt

# Define train/test data
find /image%folder%path/ -wholename "regular%expression.png" > output.txt

# Delete python cache
find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf

# *******************************************************************************

# Amazon instance
aws s3 cp --recursive /home/ubuntu/scania-raw-diff/ s3://flau33 --profile scania-ai-autonomous-dev
aws s3 cp --recursive s3:// <local_dir>/

# Remove preinstalled cudatoolkit to use pytorch
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/opt/aws-ofi-nccl/lib:/usr/local/lib:/usr/lib

# https://stackoverflow.com/questions/32500498/how-to-make-a-process-run-on-aws-ec2-even-after-closing-the-local-machine

# *******************************************************************************

# https://arxiv.org/pdf/2012.09841.pdf
# https://github.com/CompVis/taming-transformers?tab=readme-ov-file

# Train network (GPU)
python3 main.py \
        --resume /home/ubuntu/scania-raw-diff/2022-CVPR-LDM/src/embedder/logs/checkpoints/vqvae/f4_d3/rgb_rgb_r256_uint8.ckpt \
        --base   /home/ubuntu/scania-raw-diff/2022-CVPR-LDM/src/embedder/logs/configs/vqvae/vqvae_imagenet_f4_d3.yaml \
        --train True --accelerator gpu --devices 1 --max_epochs 40  --log_every_n_steps 1

# Reconstruct samples
python3 deploy.py \
        --resume /home/ubuntu/scania-raw-diff/2022-CVPR-LDM/src/embedder/logs/checkpoints/epoch=000014.ckpt \
        --outdir /home/ubuntu/scania-raw-diff/2022-CVPR-LDM/data/results \
        --config /home/ubuntu/scania-raw-diff/2022-CVPR-LDM/src/embedder/logs/configs/vqvae/vqvae_imagenet_f4_d3.yaml

python3 eval.py \
        --config /home/ubuntu/scania-raw-diff/src/embedder/logs/configs/vqvae/vqvae_imagenet_f4_d3.yaml

# *******************************************************************************

# https://arxiv.org/pdf/2112.10752.pdf
# https://github.com/CompVis/latent-diffusion/tree/main

# https://pytorch.org/docs/master/generated/torch.Tensor.normal_.html#torch.Tensor.normal_

# Train network (GPU)
python3 main.py \
        --resume /home/ubuntu/scania-raw-diff/src/generator/logs/checkpoints/vqgan/f4_d3/base_cond.ckpt \
        --base   /home/ubuntu/scania-raw-diff/src/generator/logs/configs/ddpm_vqgan_f4_d3_cond.yaml \
        --train True --accelerator gpu --devices 1 --max_epochs 80 --log_every_n_steps 1

python3 main.py \
        --base   /home/ubuntu/scania-raw-diff/src/generator/logs/configs/ddpm_vqgan_f4_d3.yaml \
        --train True --accelerator gpu --devices 4 --max_epochs 20 --log_every_n_steps 1

# Reconstruct images (uncompress)
python3 deploy.py \
        --resume /home/ubuntu/scania-raw-diff/src/generator/logs/checkpoints/epoch=000019.ckpt \
        --outdir /home/ubuntu/scania-raw-diff/data/results/ \
        --config /home/ubuntu/scania-raw-diff/src/generator/logs/configs/ddpm_vqgan_f4_d3_cond.yaml