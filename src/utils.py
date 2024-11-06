import os
import re
import cv2
import torch
import importlib
import torchvision
import numpy as np
import albumentations


from PIL import Image
from os import listdir
from einops import rearrange
from omegaconf import OmegaConf
from os.path import isfile, join
from torch.utils.data.dataloader import default_collate

from generator.src.modules.noise import degradation_bsrgan

###

def save_image(x, path):
    Image.fromarray((torch.clamp(x.detach().cpu(), -1, 1).numpy().transpose(1, 2, 0) * 255).astype(np.uint8)).save(path)


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':      return None
        elif config == "__is_unconditional__":  return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def load_model_from_config(config, state_dict, gpu=True, eval_mode=True):
    
    if "ckpt_path" in config.params:
        print("Deleting the restore-ckpt path from the config...")
        config.params.ckpt_path = None

    if "downsample_cond_size" in config.params:
        print("Deleting downsample-cond-size from the config and setting factor=0.5 instead...")
        config.params.downsample_cond_size = -1
        config.params["downsample_cond_factor"] = 0.5

    try:

        if "ckpt_path" in config.params.first_stage_config.params:
            config.params.first_stage_config.params.ckpt_path = None
            print("Deleting the first-stage restore-ckpt path from the config...")

        if "ckpt_path" in config.params.cond_stage_config.params:
            config.params.cond_stage_config.params.ckpt_path = None
            print("Deleting the cond-stage restore-ckpt path from the config...")

    except:  pass

    model = instantiate_from_config(config)

    if state_dict is not None:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Missing Keys in State Dict: {missing}")
        print(f"Unexpected Keys in State Dict: {unexpected}")

    if gpu:         model.cuda()
    if eval_mode:   model.eval()

    return {"model": model}


# https://codegolf.stackexchange.com/questions/86410/reverse-bayer-filter-of-an-image
def undebayer(img_path):
    import mat73
 
    # img = Image.open(img_path)
    # rescaler =     albumentations.SmallestMaxSize(max_size = 256)  # Max-size 1024
    # cropper =      albumentations.CenterCrop(height=256,width=256)
    # preprocessor = albumentations.Compose([rescaler, cropper])
    # img = Image.fromarray(preprocessor(image=np.array(img).astype(np.uint8))['image'])

    width, height = img.size

    raw = Image.new('RGB', (2 * width, 2 * height))
    P = raw.putpixel
    for blue in range(width * height):
        x = blue//height; y = blue%height
        red, green, blue = img.getpixel((x,y))
        
        c = 2 * x; d = 2 * y; 
        G = (0, green ,0)
        
        P((c, d), (0, 0, blue));   P((c + 1, d), G)
        P((c, d + 1), G);          P((c + 1, d + 1), (red, 0, 0))

    return raw


def preprocess_image_small(image_path, n_blocks=4):
    image = np.array(Image.open(image_path)).astype(np.uint8)
    height, width, _ = image.shape

    block_array = [];  block_size = height//2;  block_shift = 4 # 4, 128, 256

    down_block = torchvision.transforms.ToTensor()( image[                      :block_size+block_shift,  :block_size+block_shift, :])
    up_block = torchvision.transforms.ToTensor()(   image[block_size-block_shift:2*block_size,            :block_size+block_shift, :])
    block_array.append(down_block); block_array.append(up_block)

    for idx in range(1, n_blocks-1):
        down_block = torchvision.transforms.ToTensor()( image[                      :block_size+block_shift,  idx*block_size-(block_shift//2):(idx+1)*block_size+(block_shift//2), :])
        up_block = torchvision.transforms.ToTensor()(   image[block_size-block_shift:2*block_size,            idx*block_size-(block_shift//2):(idx+1)*block_size+(block_shift//2), :])
        block_array.append(down_block); block_array.append(up_block)

    down_block = torchvision.transforms.ToTensor()(     image[          :block_size+block_shift,    width-block_size-block_shift:width, :])
    up_block = torchvision.transforms.ToTensor()(       image[block_size-block_shift:2*block_size,  width-block_size-block_shift:width, :])
    block_array.append(down_block); block_array.append(up_block)

    return (block_array, [height, width, block_size, block_shift])


def preprocess_image_big(image_path, n_blocks=2):
    image = np.array(Image.open(image_path)).astype(np.uint8)
    height, width, _ = image.shape

    block_array = [];  block_size = height;  block_shift = 0

    block_array.append(torchvision.transforms.ToTensor()(image[:, :block_size+block_shift, :]).to(torch.float16))
    block_array.append(torchvision.transforms.ToTensor()(image[:, width-block_size-block_shift:width, :]).to(torch.float16))

    return (block_array, [height, width, block_size, block_shift])

### 

if False:  # Reconstruction List
    file_path = '/home/ubuntu/scania-raw-diff/src/embedder/logs/optim/rgb-raw/sets/train_rec.txt'
    folder_path = '/home/ubuntu/scania-raw-diff/src/embedder/logs/optim/rgb-raw/vqvae_f4_d3_rgb_raw_r256_e20/xrec_r512/train'
    imgs = [join(folder_path, f) for f in listdir(folder_path) if isfile(join(folder_path, f))]

    file = open(file_path, 'w')
    for e in sorted(imgs):  file.write(e + '\n')

###

if False:  # Noise Addition
    outdir = '/home/ubuntu/scania-raw-diff/src/'
    img = Image.open('')

    width, height = img.size
    left =  (width - 1024)/2;   top =   (height - 1024)/2
    right = (width + 1024)/2;  bottom = (height + 1024)/2

    img = img.crop((left, top, right, bottom))
    img = rearrange(torchvision.transforms.ToTensor()(img), 'c h w -> h w c').numpy()
    Image.fromarray((img * 255).astype(np.uint8)).save(outdir + 'noisy0.png')

    degradation = lambda x: x + np.random.normal(0, 35 / 255.0, x.shape).astype(np.float32)

    noisy = degradation(img)
    for idx in range(1, 4):
        noisy = degradation(noisy)
        Image.fromarray((noisy * 255).astype(np.uint8)).save(outdir + 'noisy' + str(idx) + '.png')

###

if False:  # Original images
    file = '/home/ubuntu/scania-raw-diff/src/embedder/logs/optim/rgb-rgb/sets/train.txt'
    outdir = '/home/ubuntu/scania-raw-diff/src/'

    img_path_array = [1]

    rescaler =     albumentations.SmallestMaxSize(max_size = 512) # Max-size 1024
    cropper =      albumentations.CenterCrop(height=512, width=512)
    preprocessor = albumentations.Compose([rescaler, cropper])

    with open(file, 'r') as descriptor:
        lines = descriptor.readlines()
        
        for img_path in img_path_array:
            img = np.array(Image.open(lines[img_path][:-1])).astype(np.uint8)
            img = Image.fromarray(preprocessor(image=img)["image"]).save(outdir + "{:08}_rgb.png".format(img_path))

            img = np.array(Image.open(lines[img_path][:-1].replace('/rgb/', '/raw/'))).astype(np.uint8)
            img = Image.fromarray(preprocessor(image=img)["image"]).save(outdir + "{:08}_raw.png".format(img_path))

### 

if False:  # Undebayering images
    file = '/home/ubuntu/scania-raw-diff/src/embedder/logs/optim/rgb-rgb/sets/train.txt'
    outdir = '/home/ubuntu/scania-raw-diff/src/'

    with open(file, 'r') as descriptor:
        lines = descriptor.readlines()
        
        for img_path in [1]:  img = undebayer(lines[img_path][:-1]).save(outdir + "{:08}.png".format(img_path))

###

if False:  # Reconstruct high-resolution images (generative autoencoders)

    config = OmegaConf.load('/home/ubuntu/scania-raw-diff/src/embedder/logs/configs/vqgan/vqgan_f8_d4.yaml')

    state_dict = torch.load('/home/ubuntu/scania-raw-diff/src/embedder/logs/checkpoints/epoch=000000.ckpt', map_location='cpu')["state_dict"]
    
    model = load_model_from_config(config.model, state_dict, gpu=True)["model"]
    
    n_blocks = 4
    x_block_array, block_values = preprocess_image_big('/home/ubuntu/scania-raw-diff/auxiliary/data/proframe_dev0_port4_2023-02-28_13-01-45/v1/00000001.png', n_blocks=n_blocks)
    height, width, block_size, block_shift = block_values

    # rescaler_small = albumentations.SmallestMaxSize(max_size = 64)
    # rescaler_normal = albumentations.SmallestMaxSize(max_size = block_size + block_shift)

    xrec_block_array = []
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        with torch.no_grad():
            for section in x_block_array:
                x = default_collate([section]).to(model.device)

                # xz = model.encode(x); xrec = model.decode(xz.sample())  # VQVAE
                xz, _, _ = model.encode(x);  xrec = model.decode(xz);   # VQGAN
                xrec_block_array.append(xrec.detach().cpu())
                del x, xz, xrec;  torch.cuda.empty_cache()

                # xrec = rescaler_small(image=rearrange(x, 'c h w -> h w c').numpy())["image"] # Linear Resizer
                # xrec = rescaler_normal(image=xrec)["image"]
                # xrec_block_array.append([torchvision.transforms.ToTensor()(xrec)])

    img = torch.zeros((3, height, width), dtype=torch.float16)

    if False:  # preprocess_image_small
        img[:, block_size:2*block_size, (n_blocks-1)*block_size:width] = xrec_block_array.pop()[0][:, block_shift:,  (block_size+block_shift) - (width-((n_blocks-1)*block_size)):block_size+block_shift]
        img[:,           :block_size,  (n_blocks-1)*block_size:width] = xrec_block_array.pop()[0][:, :-block_shift, (block_size+block_shift) - (width-((n_blocks-1)*block_size)):block_size+block_shift]

        for idx in range(n_blocks-1, 1, -1):
            img[:, block_size:2*block_size, ((idx-1)*block_size):idx*block_size] = xrec_block_array.pop()[0][:, block_shift:, block_shift//2:-(block_shift//2)]
            img[:,           :block_size,   ((idx-1)*block_size):idx*block_size] = xrec_block_array.pop()[0][:, :-block_shift, block_shift//2:-(block_shift//2)]

        img[:, block_size:2*block_size, :block_size] = xrec_block_array.pop()[0][:, block_shift:, :-block_shift]
        img[:,           :block_size,   :block_size] = xrec_block_array.pop()[0][:, :-block_shift, :-block_shift]

    if True:  # preprocess_image_big
        img[:, :, width-block_size:width] = xrec_block_array.pop()[0]
        img[:, :, :block_size] = xrec_block_array.pop()[0]

    save_image(img, os.path.join('/home/ubuntu/scania-raw-diff/src/', "{:08}.png".format(1)))

###

if False:  # Reconstruct high-resolution images (diffusion)

    from deploy import run

    config = OmegaConf.load('/home/ubuntu/scania-raw-diff/src/generator/logs/configs/ddpm_vqgan_f4_d3_cond.yaml')

    state_dict = torch.load('/home/ubuntu/scania-raw-diff/src/generator/logs/checkpoints/vqgan/f4_d3/rgb_raw_r512_e10.ckpt', map_location='cpu')["state_dict"]
    model = load_model_from_config(config.model, state_dict, gpu=True)["model"]

    x_block_array, block_values = preprocess_image_big('/home/ubuntu/scania-raw-diff/data/object-detection/0001560a-8ea1-4bc8-bcca-8bb45025a08b_anonymized/0001560a-8ea1-4bc8-bcca-8bb45025a08b_299.jpg')
    old_height, old_width, old_block_size, block_shift = block_values

    height = 1024;  ratio = height/old_height;  width = int(old_width * ratio);  block_size = int(old_block_size * ratio);  block_shift = 50

    rescaler_small = albumentations.SmallestMaxSize(max_size = height//4)
    rescaler_normal = albumentations.SmallestMaxSize(max_size = height)
    
    xrec_block_array = []
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        with torch.no_grad():
            for section in x_block_array:
                x = default_collate([section]).to(model.device)
                # xz, _, _ = model.encode(x); xrec = model.decode(xz)

                x = (rearrange(x[0], 'c h w -> h w c').cpu().numpy() * 255).astype(np.uint8)
                data_dict = dict()
                data_dict["image"] =    rearrange(torchvision.transforms.ToTensor()(rescaler_normal(image=x)["image"]), 'c h w -> h w c').unsqueeze(0)
                data_dict["LR_image"] = rearrange(torchvision.transforms.ToTensor()(rescaler_small(image=x)["image"]),  'c h w -> h w c').unsqueeze(0)

                # print(x.shape);  print(data_dict["image"].shape);  print(data_dict["LR_image"].shape)

                xrec = run(model, data_dict, custom_steps = 25)["sample"]
                xrec_block_array.append(torch.clamp(xrec.detach().cpu() , -1., 1.))
                
                del x, xrec;  torch.cuda.empty_cache()
                # del x, xz, xrec;  torch.cuda.empty_cache()

    img = torch.zeros((3, height, width), dtype=torch.float16)

    img[:, :, block_size-block_shift:width] =   xrec_block_array.pop()[0][:, :,  block_size -(width+block_shift-block_size):block_size]
    img[:, :, :block_size-block_shift] =        xrec_block_array.pop()[0][:, :, :-block_shift]

    save_image(img, os.path.join('/home/ubuntu/scania-raw-diff/src/', "{:08}.png".format(1)))

###

if False:  # Obtain metrics for specific data

    from generator.src.modules.losses.metrics import *
    # from main import WrappedDataset
    # from torchmetrics.image.fid import FrechetInceptionDistance
    from embedder.src.dataloader import CustomTrain
    from torch.utils.data import DataLoader, Dataset

    class WrappedDataset(Dataset):  # Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset
        def __init__(self, dataset):
            self.data = dataset

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    dataset = WrappedDataset(CustomTrain(256, '/efs/home/flau33/scania-frame-diff/2022-CVPR-LDM/sidd.txt'))
    dataloader = DataLoader(dataset, batch_size=1, num_workers=8, shuffle=False)

    fid_obj = FrechetInceptionDistance(feature=2048).to('cuda')

    for batch in tqdm(dataloader):
        img = (batch['image'].permute(0, 3, 1, 2) * 255).type(torch.uint8).to(torch.device("cuda"))[:, :-1, :, :]
        fid_obj.update(img, real=True)

        noisy = (batch['noisy'].permute(0, 3, 1, 2) * 255).type(torch.uint8).to(torch.device("cuda"))
        fid_obj.update(noisy, real=False)

    fid = fid_obj.compute().item()
    
    ssim = 0; psnr = 0; l2 = 0
    for batch in tqdm(dataloader):  # Expected batch size 1

        img = batch['image'][0][:, :, :-1]
        noisy = batch['noisy'][0]

        # Reconstruction Loss        
        l2 += torch.nn.functional.mse_loss(img, noisy)

        img = (img * 255).type(torch.uint8).numpy()
        noisy = (noisy * 255).type(torch.uint8).numpy()

        # Structural Similarity
        ssim += structural_similarity(img, noisy)  

        # Peak Signal to Noise ratio    
        psnr += peak_signal_noise_ratio(img, noisy)

    ssim /= len(dataloader); psnr /= len(dataloader); l2 /= len(dataloader)

    print('{0}:\tFID: {1}\tL2: {2}\tSSIM: {3}\tPSNR: {4}'.format('Metrics', fid, l2, ssim, psnr))

###

if True:  # Random crop exploration over separate images

    import glob

    root_path = '/efs/home/flau33/efs/smartphone-image-denoising-dataset/data/'
    folder_array = [f for f in os.listdir(root_path) if not os.path.isfile(f)]
    for idx_folder, folder in enumerate(folder_array, 0):

        image_path = glob.glob(root_path + folder + '/' + '*GT*.PNG')[0]  # Only one GT image
        image = np.array(Image.open(image_path))
        
        for idx_patch in range(0, 16, 1):  # 16 patches for each HR image -> 320 images are 5120 patches
            crop = albumentations.RandomCrop(height=256, width=256)(image=image)["image"]
            Image.fromarray(crop).save('/efs/home/flau33/train/' + str(idx_folder).zfill(4) + '-' + str(idx_patch).zfill(4) + '.png')