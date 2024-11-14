# Unprocessed Frame Diffusion for Learned Image Compression ([DiVA 2024](https://www.diva-portal.org/smash/record.jsf?pid=diva2%3A1871427&dswid=-5020))

## Table of Contents
1. [Introduction](#introduction)
2. [Motivation](#motivation)
3. [Theoretical Background](#theoretical-background)
    1. [About Generative-Adversarial Networks (GAN)](#about-generative-adversarial-networks-gan)
    2. [About Variational Autoencoders (VAE)](#about-variational-autoencoders-vae)
    3. [About Gaussian Diffusion](#about-gaussian-diffusion)
4. [Implementation](#implementation)
    1. [Installation (Ubuntu Focal 20.04)](#installation-ubuntu-focal-2004)
    2. [Dataset Characteristics](#dataset-characteristics)
    3. [Experimentation](#experimentation)
         1. [Embedding Architecture](#embedding-architecture)
         2. [Generative Architecture](#generative-architecture)

## Introduction 

Image processing is an engineering field responsible for the digitalization of images and their corresponding manipulations e.g. color correction, deblurring, feature extraction, etc. Image processing considers images as high dimensional signals represented with 2D camera samplings of discrete values.
Image digitalization is performed with an operational pipeline that transforms the raw signal data from a CMOS1 sensor onto discrete colored images. This pipeline is commonly known as Image Signal Processor (ISP).
 
Digitalized images exist with different formats depending on the purpose, e.g. configurations for storage or transmission. These formats are levels of compression from the unprocessed data that remove information to reduce the size of the representation and provide visually pleasing images. However, traditional compression algorithms lose pixel values that are meaningful for feature extraction or the semantic meaning of the image signal.
 
Neural Network (NN) architectures have achieved super-human performance on tasks that include non-linear mappings of high-dimensional data. NN-based feature extraction techniques appeared within the Machine Learning (ML) engineering field as Representation Learning (RL).

RL succeeds at identifying the uniqueness of an image in terms of features such as shape, color, pixel correlation or semantic meaning. These features are latent continuous representations, hardly explainable by humans, that are obtained from the output of an NN model.
State-Of-The-Art (SOTA) models rely on autoencoder architectures, composed of both encoder and decoder. The downscaling and upscaling of the feature extraction process with autoencoders seems like a natural extension of the compression paradigm.

## Motivation

Generative models have shown powerful capabilities in learning fidelic image feature distributions. However, the parameter space of these distributions is high-dimensional and random sampling is prone to high-variance results.
 
The likelihood formulation, allows a conditioning over the parameter space. In practice, this conditioning is performed with other modes of data such as text or semantic maps. The result of this process is a marginalization of the parameter space that allows targeted generations from random samples.
Unprocessed CMOS data can be subject to noise and redundant values. However, leveraging RL on this type of data is expected to give better meaningful features than RGB processed images for color, light intensity and semantic meaning.
 
Techniques such as multihead attention, that focuses on similar areas of the data distribution for feature extraction, may take advantage of the redundant information. Moreover, this technique displays outstanding results on text, audio and image feature extraction.
 
This research explores the effect and performance of unprocessed image features as compressed representations and conditioning elements in the image generation process of diffusion models.

## Theoretical Background

### About Generative-Adversarial Networks (GAN)

GANs are based on the minimax principle of game theory. Having a player and an opponent competing over a game, the minimax value of a player is the smallest value that the opponent can force the player to receive and the largest value the player can assure. Mathematically speaking, the minimax value is the player’s gain minimized by the opponent and maximized by the player.

The neural network interpretation trains two models simultaneously: (1) A generative model G(z) captures the data distribution and (2) a discriminative model D(x) estimates the probability that a sample came from the training data x rather than G(z).

In this scenario, the discriminator would act as the player, trying to maximize the likelihood log D(x) of assigning the correct label to both training examples and samples from G(z). The generator would act as the opponent, trying to minimize log (1−D(G(z))) i.e. reducing the likelihood of identifying the generated sample as such.

The Vector-Quantized GAN is trained to maximize the log-likelihood while imposing the prior distribution. However, the discriminator network subjects this prior by providing error gradients over the latent embedding being a correct or incorrect prior sample.

<p align="center">
  <img src="https://github.com/user-attachments/assets/4f78721f-1915-454d-8240-e52b184c1b8f" width="500"/>
</p>

### About Variational Autoencoders (VAE)

VAEs leverage a neural network auto-encoding architecture (i.e. encoder and decoder) to approximate the posterior distribution P(z|x) defining the model parameters z that satisfy the observed data x. The training process differentiates and optimizes the lower bound of the log-likelihood w.r.t. the variational parameters ϕ of the encoder Qϕ (posterior distribution approximation) and the generative parameters θ of the decoder Pθ (conditional distribution).

The encoding output can be thought as the high-dimensional parameters (mean µ and variance σ2) of the posterior distribution approximation. Random sampling from the learned distribution is used to obtain z and train the decoder. This step is non-differentiable and does not allow backpropagation.  

In practice, (1) the learned variance is applied to the sampled noise and (2) the learned mean is added, resulting in a random sample of the distribution that can be decoded.

The Vector-Quantized VAE is trained to maximize the lower bound of the posterior distribution loglikelihood while constraining the encoder over a prior distribution. The training objective uses the KL divergence to impose the prior and a reconstruction error such as MSE over the original and reconstructed images to optimize the autoencoder towards the real posterior.

<p align="center">
  <img src="https://github.com/user-attachments/assets/872a0d2d-d6a4-4ed3-aa2d-75cd5d24ba56" width="400"/>
</p>

### About Gaussian Diffusion

Diffusion is a thermodynamics concept in which elements from high density areas move to low density areas. Generative models can make use of this process by transforming one distribution PX (e.g. data distribution) into another distribution PZ (e.g. Gaussian distribution) by accumulating small perturbations (i.e. noise addition) at a defined β rate.

These perturbations are modelled by Markov chains with a Gaussian target distribution. The noisy sample is forwarded to an additional process of reverse diffusion (denoising) modelled by a neural network whose objective is to learn the previously applied perturbations.

<p align="center">
  <img src="https://github.com/user-attachments/assets/560f8e6b-544f-4238-a780-f1cbee212e91" width="700"/>
</p>

## Implementation

### Installation (Ubuntu Focal 20.04)

#### Requirements and Dependencies

```
# Install dependencies
pip install -r ~/scania-thesis-work/config/requirements.txt

# ROS Installation
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-get -y install curl # if you haven't already installed curl
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo apt-get update;  sudo apt-get -y install ros-noetic-desktop
source /opt/ros/noetic/setup.bash

# GCC Compiler
sudo apt-get -y install gcc c++; gcc --version;  g++ --version

# CUDA Toolkit (-L/usr/lib/wsl/lib/)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4

# Solectrix
sudo apt-get -y install libsdl2-ttf-dev libgl-dev libglm-dev libavcodec-dev libavformat-dev libavutil-dev libjpeg-turbo8-dev libpng-dev libswscale-dev libopencv-dev libsdl2-dev
PATH=/home/ubuntu/scania-raw-diff/data/solectrix/src/release/:$PATH

# SoftISP
sudo apt-get install ffmpeg
sudo dpkg -i /home/ubuntu/scania-raw-diff/data/soft-isp/231024_scania-man_SoftISP_3.8.2_ubuntu_20.04.deb
sudo apt-get -f install  # Dependencies installation
SoftISP --verbose=1 \
        --input rosbag \
        --file /home/ubuntu/campipe-road-data/imx490/221103/proframe_dev0_port16_2022-11-03_10-16-48.bag \
        --config /home/ubuntu/scania-raw-diff/data/soft-isp/profiles/imx490.json
```

#### Data Indexing

```
# Define train/test data
find /image%folder%path/ -wholename "regular%expression.png" > output.txt
```

#### Embedding Architecture Training and Deployment

Examples of model configuration files can be found under [*/src/embedder/docs*](/src/embedder/docs/).

```
# https://arxiv.org/pdf/2012.09841.pdf
# https://github.com/CompVis/taming-transformers?tab=readme-ov-file

# Train network (GPU)
python3 main.py \
        --resume /checkpoint%path --base /model%config%path \
        --train True --accelerator gpu --devices 1 --max_epochs 40 --log_every_n_steps 1

# Reconstruct samples
python3 deploy.py \
        --resume /checkpoint%path --outdir /results%directory%path --config /model%config%path

python3 eval.py \
        --config /model%config%path
```

#### Generative Architecture Training and Deployment

Examples of model configuration files can be found under [*/src/generator/docs*](/src/generator/docs/).

```
# https://arxiv.org/pdf/2112.10752.pdf
# https://github.com/CompVis/latent-diffusion/tree/main

# Fine-tune or Train network (GPU)
python3 main.py \
        --resume /checkpoint%path --base /model%config%path \
        --train True --accelerator gpu --devices 1 --max_epochs 80 --log_every_n_steps 1

python3 main.py \
        --base  /model%config%path --train True --accelerator gpu --devices 4 --max_epochs 20 --log_every_n_steps 1

# Reconstruct images (uncompress)
python3 deploy.py \
        --resume /checkpoint%path --outdir /results%directory%path --config /model%config%path
```

### Dataset Characteristics

The dataset consists of an unprocessed image collection of 2676 elements stored in CSI-2 format from three different types of cameras with a respective (width x height) resolution of (1936x1112), (2896x1876) and (3868x2176). The images are collected from an onboard sensor vehicle during real time while driving around the company’s facilities.

The images have been pre-processed through the company’s ISP software to obtain the RGB counterparts for every unprocessed sample, thus creating a paired RAW-RGB dataset. This dataset is of essential importance to understand the distribution shift imposed by the overhead of visually pleasing transformations over RGB images.

The RGB samples are used to fine-tune the models and consider the resulting networks as baselines for the RAW evaluation metrics. Moreover, the RAW samples are embedded onto a representation used as conditioning for the image generation process.

The dataset has been divided onto train, validation and test sets with respective ratios of (0.6, 0.2, 0.2). Pre-processing of the data before optimization includes normalization on the [0, 1] range, rescaling to (512x512) resolution and performing center crops.

<table>
<tr>
<td><img src="https://github.com/user-attachments/assets/b56db771-51a0-4c5b-aa6a-004892700a90"/></td>
<td><img src="https://github.com/user-attachments/assets/cddd9f5f-afcb-487e-a706-9057e7e514cf"/></td>
</tr>
<tr>
<td>
<p align='Center'>RGB Image</p>
</td>
<td>
<p align='Center'>RAW Image</p>
</td>
</tr>
</table>

### Experimentation

#### Embedding Architecture

The purpose of the embedder is to perform linear or non-linear transformations of the data to lower dimension representations while retaining the defining components of the input. The latent space or parameter space of the embedding aids the generation process by excluding redundant or low variance values, allowing the decoder architecture to reconstruct the input and obviate encoder-removed values from the learned input distributions.

The architecture chosen is characterized by two components: (1) An autoencoder that embeds and reconstructs the data and (2) a codebook that vector-quantizes or discretizes the possible values of the embedding (i.e. the posterior approximation).

Two open-sourced implementations have been chosen as embedders. Inspired respectively by VAEs and GANs, these architectures rivalled in the generative field previous to the usage of Gaussian diffusion.

#### Generative Architecture

The purpose of the generator is to complete additional information considered redundant or non-descriptive from the input image. This information is not available from the embedding (non-characteristic features), but is learned as an element of the input distribution. Distributing feature learning over different components, allows improved optimization and better representation of the real posterior.

  1. A feature embedder learning the representation of the data provides local context.
  2. A learned codebook provides global context when the features are vector-quantized.
  3. A learned backwards denoiser refines the latent features towards the real posterior.

The generative architecture used, namely Latent Diffusion, fixes the encoder as a Markov chain of noise additions and optimizes the decoder to learn the transition matrices between noisy states. High dimensionality of the data produces consuming times. Working with latent spaces would not only benefit from reduced computations, but would also help onto distributing the RL problem onto separately optimizable networks

The fixed encoder or forward diffusion, transforms the input image onto random noise from a Gaussian distribution. The decoder transforms randomly sampled noise towards the posterior distribution. The decoder or backwards denosing, is composed by a convolutional UNet-style section that leverages multihead cross-attention to remove noise. The research contribution includes conditioning mechanisms that allow the marginalization of the parameter space over the condition data.

This is achieved by concatenating the conditioning embedding at different attended steps of the generation. An additional task of multimodal inference is delegated onto the generative architecture, i.e. using different types of data such as images, text or semantic maps simultaneously for the purpose of image generation.

<p align="center">
  <img src="https://github.com/user-attachments/assets/2c6a9f52-f312-4dbc-be90-5ad48ca4ecda" width="600"/>
</p>
