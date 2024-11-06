import cv2
import PIL
import torch
import numpy as np
import torchvision
import albumentations
import torchvision.transforms.functional as TF

from PIL import Image
from einops import rearrange
from functools import partial
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from embedder.src.dataloader import CustomBase
from generator.src.utils import instantiate_from_config
from generator.src.modules.noise import degradation_bsrgan


###

class ImagePaths_RGB_RGB(Dataset):
    def __init__(self, paths, size=None, degradation=None, downscale_f=4, random_crop=False):
        """
        Custom Superresolution Dataloader
        Performs following ops in order:
        1.  Crops a crop of size s from image either as random or center crop
        2.  Resizes crop to size with cv2.area_interpolation
        3.  Degrades resized crop with degradation_fn

        size:           Resizing to size after cropping
        degradation:    Degradation_fn, e.g. cv_bicubic or bsrgan_light
        downscale_f:    Low Resolution Downsample factor
        min_crop_f:     Determines crop size s, where s = c * min_img_side_len with c sampled from interval (min_crop_f, max_crop_f)
        """

        assert size;  assert (size / downscale_f).is_integer()

        self.size = size
        self.labels = dict()
        self._length = len(paths)
        self.labels["file_path_"] = paths
        self.center_crop = not random_crop
        self.LR_size = int(size / downscale_f)
        self.pil_interpolation = False # Gets reset later if interp_op is from pillow
        self.image_rescaler = albumentations.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_LINEAR)
        self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
        self.preprocessor = albumentations.Compose([self.image_rescaler, self.cropper])

        if degradation == "bsrgan":
            self.degradation_process = partial(degradation_bsrgan, sf=downscale_f)
        else:
            interpolation_fn = {"cv_nearest": cv2.INTER_NEAREST,    "cv_bilinear": cv2.INTER_LINEAR,
                                "cv_bicubic": cv2.INTER_CUBIC,      "cv_area": cv2.INTER_AREA,
                                "cv_lanczos": cv2.INTER_LANCZOS4,
                                "pil_nearest": PIL.Image.NEAREST,   "pil_bilinear": PIL.Image.BILINEAR,
                                "pil_bicubic": PIL.Image.BICUBIC,   "pil_box": PIL.Image.BOX,
                                "pil_hamming": PIL.Image.HAMMING,   "pil_lanczos": PIL.Image.LANCZOS,}[degradation]

            self.pil_interpolation = degradation.startswith("pil_")
            self.degradation_process = partial(TF.resize, size=self.LR_size, interpolation=interpolation_fn) if self.pil_interpolation else \
                                       albumentations.SmallestMaxSize(max_size=self.LR_size, interpolation=interpolation_fn)


    def __len__(self):
        return self._length


    def __getitem__(self, i):
        data_dict = dict()
        image = Image.open(self.labels["file_path_"][i].replace('clean', 'noisy')) # .replace('/rgb/', '/raw/')
        # target = Image.open(self.labels["file_path_"][i])

        if not image.mode == "RGB": image = image.convert("RGB")

        # Transformations of data to avoid artifacts in output image
        # (<IMG> / 127.5 - 1.0).astype(np.float32)
        image = np.array(image).astype(np.uint8)
        # target = np.array(target).astype(np.uint8)

        image = self.preprocessor(image=image)["image"]
        # target = self.preprocessor(image=target)["image"]

        LR_image = Image.fromarray(image).resize((self.LR_size, self.LR_size))
        image = (image / 127.5 - 1.0).astype(np.float32)
        # target = (target / 127.5 - 1.0).astype(np.float32)

        data_dict["image"] = rearrange(torchvision.transforms.ToTensor()(image), 'c h w -> h w c')
        # data_dict["target"] = rearrange(torchvision.transforms.ToTensor()(target), 'c h w -> h w c')
        data_dict["LR_image"] = 2. * rearrange(torchvision.transforms.ToTensor()(LR_image), 'c h w -> h w c') - 1
        data_dict["path"] = self.labels["file_path_"][i].split('/')[-1] # [-2] + ".png"

        return data_dict

###

class ImagePaths_RGB_RAW(Dataset):
    def __init__(self, paths, size=None, degradation=None, downscale_f=4, random_crop=False):
        """
        Custom Superresolution Dataloader
        Performs following ops in order:
        1.  Crops a crop of size s from image either as random or center crop
        2.  Resizes crop to size with cv2.area_interpolation
        3.  Degrades resized crop with degradation_fn

        size:           Resizing to size after cropping
        degradation:    Degradation_fn, e.g. cv_bicubic or bsrgan_light
        downscale_f:    Low Resolution Downsample factor
        min_crop_f:     Determines crop size s, where s = c * min_img_side_len with c sampled from interval (min_crop_f, max_crop_f)
        """

        assert size;  assert (size / downscale_f).is_integer()
        
        self.size = size
        self.labels = dict()
        self._length = len(paths)
        self.labels["file_path_"] = paths
        self.center_crop = not random_crop
        self.LR_size = int(size / downscale_f)
        self.pil_interpolation = False # Gets reset later if interp_op is from pillow
        
        self.rescaler =     albumentations.SmallestMaxSize(max_size = self.size, interpolation=cv2.INTER_LINEAR) # Max-size 1024
        self.cropper =      albumentations.CenterCrop(height=self.size,width=self.size)
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

        if degradation == "bsrgan":
            self.degradation_process = partial(degradation_bsrgan, sf=downscale_f)
        else:
            interpolation_fn = {"cv_nearest": cv2.INTER_NEAREST,    "cv_bilinear": cv2.INTER_LINEAR,
                                "cv_bicubic": cv2.INTER_CUBIC,      "cv_area": cv2.INTER_AREA,
                                "cv_lanczos": cv2.INTER_LANCZOS4,
                                "pil_nearest": PIL.Image.NEAREST,   "pil_bilinear": PIL.Image.BILINEAR,
                                "pil_bicubic": PIL.Image.BICUBIC,   "pil_box": PIL.Image.BOX,
                                "pil_hamming": PIL.Image.HAMMING,   "pil_lanczos": PIL.Image.LANCZOS,}[degradation]

            self.pil_interpolation = degradation.startswith("pil_")
            self.degradation_process = partial(TF.resize, size=self.LR_size, interpolation=interpolation_fn) if self.pil_interpolation else \
                                       albumentations.SmallestMaxSize(max_size=self.LR_size, interpolation=interpolation_fn)


    def __len__(self):
        return self._length


    def __getitem__(self, i):
        data_dict = dict()
        img = Image.open(self.labels["file_path_"][i])

        img = np.array(img).astype(np.uint8)
        img = self.preprocessor(image=img)["image"]

        LR_image = self.degradation_process(image=img)["image"]

        data_dict["image"] = rearrange(torchvision.transforms.ToTensor()(img), 'c h w -> h w c')
        data_dict["LR_image"] = rearrange(torchvision.transforms.ToTensor()(LR_image), 'c h w -> h w c')

        return data_dict

###

class ImagePaths_RAW_RGB(Dataset):  # TODO Update as RGB_RAW
    def __init__(self, paths, size=None, degradation=None, downscale_f=4, random_crop=False):
        """
        Custom Superresolution Dataloader
        Performs following ops in order:
        1.  Crops a crop of size s from image either as random or center crop
        2.  Resizes crop to size with cv2.area_interpolation
        3.  Degrades resized crop with degradation_fn

        size:           Resizing to size after cropping
        degradation:    Degradation_fn, e.g. cv_bicubic or bsrgan_light
        downscale_f:    Low Resolution Downsample factor
        min_crop_f:     Determines crop size s, where s = c * min_img_side_len with c sampled from interval (min_crop_f, max_crop_f)
        """

        assert size;  assert (size / downscale_f).is_integer()
        
        self.size = size
        self.labels = dict()
        self._length = len(paths)
        self.labels["file_path_"] = paths
        self.center_crop = not random_crop
        self.LR_size = int(size / downscale_f)
        self.pil_interpolation = False # Gets reset later if interp_op is from pillow
        self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
        self.image_rescaler = albumentations.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_LINEAR)
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

        if degradation == "bsrgan":
            self.degradation_process = partial(degradation_bsrgan, sf=downscale_f)
        else:
            interpolation_fn = {"cv_nearest": cv2.INTER_NEAREST,    "cv_bilinear": cv2.INTER_LINEAR,
                                "cv_bicubic": cv2.INTER_CUBIC,      "cv_area": cv2.INTER_AREA,
                                "cv_lanczos": cv2.INTER_LANCZOS4,
                                "pil_nearest": PIL.Image.NEAREST,   "pil_bilinear": PIL.Image.BILINEAR,
                                "pil_bicubic": PIL.Image.BICUBIC,   "pil_box": PIL.Image.BOX,
                                "pil_hamming": PIL.Image.HAMMING,   "pil_lanczos": PIL.Image.LANCZOS,}[degradation]

            self.pil_interpolation = degradation.startswith("pil_")
            self.degradation_process = partial(TF.resize, size=self.LR_size, interpolation=interpolation_fn) if self.pil_interpolation else \
                                       albumentations.SmallestMaxSize(max_size=self.LR_size, interpolation=interpolation_fn)


    def __len__(self):
        return self._length


    def __getitem__(self, i):
        data_dict = dict()
        image = Image.open(self.labels["file_path_"][i])
        raw_image = Image.open(self.labels["file_path_"][i].replace('rgb', 'raw'))

        if not image.mode == "RGB": image = image.convert("RGB")

        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        LR_image = self.degradation_process(image=image)["image"]

        raw_image = self.preprocessor(image=raw_image)["image"]
        raw_image = np.array(raw_image).astype(np.uint8)

        data_dict["image"] = rearrange(torchvision.transforms.ToTensor()(raw_image), 'c h w -> h w c')
        data_dict["LR_image"] = rearrange(torchvision.transforms.ToTensor()(LR_image), 'c h w -> h w c')

        return data_dict

###

class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file, **kwargs):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths_RGB_RGB(paths, size, **kwargs)
        # self.data = ImagePaths_RGB_RAW(paths, size, **kwargs)
        # self.data = ImagePaths_RAW_RGB(paths, size, **kwargs)

###

class CustomVal(CustomBase):
    def __init__(self, size, validation_images_list_file, **kwargs):
        super().__init__()
        with open(validation_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths_RGB_RGB(paths, size, **kwargs)
        # self.data = ImagePaths_RGB_RAW(paths, size, **kwargs)
        # self.data = ImagePaths_RAW_RGB(paths, size, **kwargs)

###

class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file, **kwargs):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths_RGB_RGB(paths, size, **kwargs)
        # self.data = ImagePaths_RGB_RAW(paths, size, **kwargs)
        # self.data = ImagePaths_RAW_RGB(paths, size, **kwargs)