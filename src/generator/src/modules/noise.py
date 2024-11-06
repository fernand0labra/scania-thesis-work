import numpy as np
import scipy.stats as ss
import cv2, scipy, random

from scipy import ndimage
from scipy.linalg import orth
from scipy.interpolate import interp2d

import generator.src.utils as utils

###

def mean_gaussian_kernel(mean, cov, size=15):
    center = size / 2.0 + 0.5
    kernel = np.zeros([size, size])
    
    for y in range(size):
        for x in range(size):
            cx = x - center + 1;  cy = y - center + 1
            kernel[y, x] = ss.multivariate_normal.pdf([cx, cy], mean=mean, cov=cov)

    return kernel / np.sum(kernel)

###

# https://github.com/ronaldosena/imagens-medicas-2/blob/40171a6c259edec7827a6693a93955de2bd39e76/Aulas/aula_2_-_uniform_filter/matlab_fspecial.py
def gaussian_kernel(hsize, std):
    size = (hsize - 1.0) / 2.0
    [x, y] = np.meshgrid(np.arange(-size, size + 1), np.arange(-size, size + 1))

    h = np.exp(-(x * x + y * y) / (2 * std * std))
    h[h < np.finfo(float).eps * h.max()] = 0
    
    sumh = h.sum()
    return h / sumh if sumh != 0 else h

###

def gaussian_blur(img, sf=4):
    wd2 = 4.0 + sf
    wd = 2.0 + 0.2 * sf
    
    if random.random() < 0.5:  # Generate an anisotropic Gaussian kernel
        l1 = wd2 * random.random();  l2 = wd2 * random.random()  # Scaling of eigenvalues
        theta = random.random() * np.pi  # Rotation angle range

        v = np.dot(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array([1., 0.]))
        V = np.array([[v[0], v[1]], [v[1], -v[0]]])
        D = np.array([[l1, 0], [0, l2]])

        kernel = mean_gaussian_kernel(mean=[0, 0], cov=np.dot(np.dot(V, D), np.linalg.inv(V)), size=2 * random.randint(2, 11) + 3)
    else:
        kernel = gaussian_kernel(hsize=2 * random.randint(2, 11) + 3, std=wd * random.random())

    return ndimage.filters.convolve(img, np.expand_dims(kernel, axis=2), mode='mirror')

###

def shift_pixel(x, scale, upper_left=True):  # Shift pixel for super-resolution with different scale factors
    h, w = x.shape[:2]
    shift = (scale - 1) * 0.5
    xv, yv = np.arange(0, w, 1.0), np.arange(0, h, 1.0)

    x1 = np.clip(xv + (shift if upper_left else -shift), 0, w - 1)
    y1 = np.clip(yv + (shift if upper_left else -shift), 0, h - 1)

    if x.ndim == 2:  x = interp2d(xv, yv, x)(x1, y1)
    if x.ndim == 3:
        for i in range(x.shape[-1]):
            x[:, :, i] = interp2d(xv, yv, x[:, :, i])(x1, y1)

    return x

###

def add_gaussian_noise(img, noise_level1=2, noise_level2=25):
    rnum = np.random.rand()
    noise_level = random.randint(noise_level1, noise_level2)
    
    if rnum > 0.6:    img = img + np.random.normal(0, noise_level / 255.0, img.shape).astype(np.float32)            # Add color Gaussian noise
    elif rnum < 0.4:  img = img + np.random.normal(0, noise_level / 255.0, (*img.shape[:2], 1)).astype(np.float32)  # Add grayscale Gaussian noise

    else:
        L = noise_level2 / 255.;  D = np.diag(np.random.rand(3));  U = orth(np.random.rand(3, 3))
        conv = np.dot(np.dot(np.transpose(U), D), U)
        img += np.random.multivariate_normal([0, 0, 0], np.abs(L ** 2 * conv), img.shape[:2]).astype(np.float32)   # Add  noise

    return np.clip(img, 0.0, 1.0)

###

def add_compression_noise(img):  # JPEG effect
    img = cv2.cvtColor(utils.single2uint(img), cv2.COLOR_RGB2BGR)
    img = cv2.imdecode(cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(30, 95)])[1], 1)
    return cv2.cvtColor(utils.uint2single(img), cv2.COLOR_BGR2RGB)

###

def random_crop(lq, hq, sf=4, lq_patchsize=64):
    rnd_lq_h = random.randint(0, lq.shape[0] - lq_patchsize)
    rnd_lq_w = random.randint(0, lq.shape[1] - lq_patchsize)
    
    rnd_hq_h = int(rnd_lq_h * sf)
    rnd_hq_w = int(rnd_lq_w * sf)

    return lq[rnd_lq_h:rnd_lq_h + lq_patchsize     , rnd_lq_w:rnd_lq_w + lq_patchsize     , :], \
           hq[rnd_hq_h:rnd_hq_h + lq_patchsize * sf, rnd_hq_w:rnd_hq_w + lq_patchsize * sf, :]

###

def degradation_bsrgan(image, sf=4, isp_model=None, output=False):
    """
    "Designing a Practical Degradation Model for Deep Blind Image Super-Resolution"
    ----------
    input:   (h w c),                               size: > (lq_patchsize x sf) x (lq_patchsize x sf),       range: [0, 1]
    output:  lq: low-quality patch,                 size: (lq_patchsize x lq_patchsize x C),                 range: [0, 1]
             hq: corresponding high-quality patch,  size: (lq_patchsize x sf) x (lq_patchsize x sf) x c,     range: [0, 1]
    """

    image = utils.uint2single(image)
    jpeg_prob, scale2_prob = 0.9, 0

    h1, w1 = image.shape[:2]
    image = image.copy()[:w1 - w1 % sf, :h1 - h1 % sf, ...]

    if output:  return {"image":utils.single2uint(add_gaussian_noise(image, noise_level1=2, noise_level2=25))}

    if sf == 4 and random.random() < scale2_prob:  # Downsample v1
        image = cv2.resize(image, (int(1 / 2 * image.shape[1]), int(1 / 2 * image.shape[0])), interpolation=random.choice([1, 2, 3])) if np.random.rand() < 0.5 else \
                utils.imresize_np(image, 1 / 2, True)
        image = np.clip(image, 0.0, 1.0)
        sf = 2

    shuffle_order = random.sample(range(7), 7)
    idx1, idx2 = shuffle_order.index(2), shuffle_order.index(3)
    if idx1 > idx2:  # Keep downsample v3 last
        shuffle_order[idx1], shuffle_order[idx2] = shuffle_order[idx2], shuffle_order[idx1]

    for i in shuffle_order:
        if   i == 0:      image = gaussian_blur(image, sf=sf)
        elif i == 1:      image = gaussian_blur(image, sf=sf)
        elif i == 2:
            
            if random.random() < 0.75:  # Downsample v2
                sf1 = random.uniform(1, 2 * sf)
                image = cv2.resize(image, (int(1 / sf1 * image.shape[1]), int(1 / sf1 * image.shape[0])), interpolation=random.choice([1, 2, 3]))
            else:
                k_shifted = shift_pixel(gaussian_kernel(25, random.uniform(0.1, 0.6 * sf)), sf)
                k_shifted = k_shifted / k_shifted.sum()  # Blur with shifted kernel

                image = ndimage.filters.convolve(image, np.expand_dims(k_shifted, axis=2), mode='mirror')[0::sf, 0::sf, ...]  # Nearest downsampling
            
            image = np.clip(image, 0.0, 1.0)

        elif i == 3:  # Downsample v3
            image = cv2.resize(image, (int(1 / sf * image.shape[1]), int(1 / sf * image.shape[0])), interpolation=random.choice([1, 2, 3]))
            image = np.clip(image, 0.0, 1.0)

        elif i == 4:  # Add Gaussian noise
            image = add_gaussian_noise(image, noise_level1=2, noise_level2=25)

        elif i == 5:  # Add compression noise
            if random.random() < jpeg_prob:
                image = add_compression_noise(image)

    return {"image":utils.single2uint(add_compression_noise(image))}