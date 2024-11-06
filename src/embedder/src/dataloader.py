import os
import cv2
import torch
import torchvision
import numpy as np
import albumentations

from PIL import Image
from einops import rearrange
from torch.utils.data import Dataset

###

class ImagePaths(Dataset):
    def __init__(self, paths, size=None, random_crop=False, labels=None):

        self.size = size
        self._length = len(paths)
        self.random_crop = random_crop
        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths

        # https://albumentations.ai/docs/api_reference/full_reference/?h=randomcrop#albumentations.augmentations.crops.transforms.RandomCrop
        if self.size is not None and self.size > 0:
            self.rescaler =     albumentations.SmallestMaxSize(max_size = self.size) # Max-size 1024
            self.cropper =      albumentations.CenterCrop(height=self.size,width=self.size) if not self.random_crop else \
                                albumentations.RandomCrop(height=self.size,width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs


    def __len__(self):
        return self._length
            

    def preprocess_image(self, image_path, dtype="uint8", h_rand_start=np.random.rand(), w_rand_start=np.random.rand()):

        if dtype == "uint8":
            image = np.array(Image.open(image_path)).astype(np.uint8)

        elif dtype == "uint8as16":
            image = np.array(Image.open(image_path)).astype(np.uint16) * 2**8

        elif dtype == "uint16":
            image = np.array(cv2.imread(image_path, cv2.IMREAD_UNCHANGED)).astype(np.uint16)

        # image = albumentations.random_crop(image, crop_height=self.size, crop_width=self.size, h_start=h_rand_start, w_start=w_rand_start)

        # image = self.preprocessor(image=image)["image"]
        # return rearrange((torchvision.transforms.ToTensor()(image).to(torch.float32) / 2**16), 'c h w -> h w c')
        return rearrange((torchvision.transforms.ToTensor()(image)), 'c h w -> h w c')


    def __getitem__(self, i):
        data_dict = dict()      

        if False:  # MPRNet
            noisy_path = self.labels["file_path_"][i].replace('2020-CVPR-CycleISP/datasets/sidd/sidd_rgb/clean', '2021-CVPR-MPRNet/Denoising/Datasets/results/sidd/sidd_rgb')
        
            file_name_start = str(int(noisy_path.split('/')[-1].split('-')[0][2:]) + 1).zfill(4)
            file_name_end = str(int(noisy_path.split('/')[-1].split('-')[-1][2:4]) + 1).zfill(2) + '.png'

            noisy_path = os.path.join('/', *noisy_path.split('/')[:-1], file_name_start + '_' + file_name_end)
            data_dict["noisy"] = self.preprocess_image(noisy_path, "uint8")

        if False:  # MSANet
            data_dict["noisy"] = self.preprocess_image(self.labels["file_path_"][i].replace('2020-CVPR-CycleISP/datasets/sidd/sidd_rgb/clean', '2022-NeurIPS-MSANet/results/sidd/sidd_rgb'), "uint8")

            file_name = str(int(noisy_path.split('/')[-1].split('-')[0]) * 32 + int(noisy_path.split('/')[-1].split('-')[-1][2:4]) + 1).zfill(4) + '_s1_x1_SR.png'

            noisy_path = os.path.join('/', *noisy_path.split('/')[:-1], file_name)
            data_dict["noisy"] = self.preprocess_image(noisy_path, "uint8")

        if True:  # Ours
            data_dict["noisy"] = self.preprocess_image(self.labels["file_path_"][i])

        h_rand_start = np.random.rand()
        w_rand_start = np.random.rand()

        data_dict["image"] = self.preprocess_image(self.labels["file_path_"][i].replace('results/noise/recon', 'sidd/sidd_raw/clean'), h_rand_start=h_rand_start, w_rand_start=w_rand_start)
        # data_dict["target"] = self.preprocess_image(self.labels["file_path_"][i], h_rand_start=h_rand_start, w_rand_start=w_rand_start)
        # data_dict["path"] = self.labels["file_path_"][i].split('/')[-1]

        for k in self.labels:  data_dict[k] = self.labels[k][i]
        return data_dict

###

class NumpyPaths(ImagePaths):
    def preprocess_image(self, image_path):

        image = np.transpose(np.load(image_path).squeeze(0), (1,2,0))
        image = np.array(Image.fromarray(image, mode="RGB")).astype(np.uint8)

        image = self.preprocessor(image=image)["image"]

        return rearrange(torchvision.transforms.ToTensor()(image), 'c h w -> h w c')

###
    
class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

###

class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)

###

class CustomVal(CustomBase):
    def __init__(self, size, validation_images_list_file):
        super().__init__()
        with open(validation_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)

###

class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)