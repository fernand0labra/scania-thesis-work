# https://github.com/twhui/SRGAN-pyTorch
# https://github.com/xinntao/BasicSR

import importlib
import numpy as np
import multiprocessing as mp
import os, cv2, math, time, torch

from queue import Queue
from collections import abc
from threading import Thread
from inspect import isfunction
from PIL import Image, ImageDraw, ImageFont

###

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif']


###

def log_txt_as_img(shape, caption_array, size=10):
    txt_array = list()
    nc = int(40 * (shape[0] / 256))
    
    for idx in range(len(caption_array)):
        txt = Image.new("RGB", shape, color="white")
        lines = "\n".join(caption_array[idx][start:start + nc] for start in range(0, len(caption_array[idx]), nc))

        try:                        ImageDraw.Draw(txt).text((0, 0), lines, fill="black", font=ImageFont.truetype('data/DejaVuSans.ttf', size=size))
        except UnicodeEncodeError:  print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txt_array.append(txt)

    return torch.tensor(np.stack(txt_array))

###

def ismap(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)

###

def isimage(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)

###

def exists(x):
    return x is not None

###

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

###

def mean_flat(tensor):
    """
    https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/nn.py#L86
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

###

def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params

###

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':      return None
        elif config == "__is_unconditional__":  return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

###

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

###

def _do_parallel_data_prefetch(func, Q, data, idx, idx_to_fn=False):
    Q.put([idx, func(data, worker_id=idx) if idx_to_fn else func(data)])
    Q.put("Done")

###

def parallel_data_prefetch(func: callable, data, n_proc, target_data_type="ndarray", cpu_intensive=True, use_worker_id=False):
    if isinstance(data, np.ndarray) and target_data_type == "list":
        raise ValueError("list expected but function got ndarray.")
    
    elif isinstance(data, abc.Iterable):
        if isinstance(data, dict):
            print(f'WARNING:"data" argument passed to parallel_data_prefetch is a dict: Using only its values and disregarding keys.')
            data = list(data.values())
        
        data = np.asarray(data) if target_data_type == "ndarray" else list(data)

    else:
        raise TypeError(f"The data, that shall be processed parallel has to be either an np.ndarray or an Iterable, but is actually {type(data)}.")

    if cpu_intensive:  Q = mp.Queue(1000);  proc = mp.Process
    else:              Q = Queue(1000);     proc = Thread

    # Spawn processes
    if target_data_type == "ndarray":
        arguments = [[func, Q, part, i, use_worker_id] 
                     for i, part in enumerate(np.array_split(data, n_proc))]
    
    else:
        step = (int(len(data) / n_proc + 1) if len(data) % n_proc != 0 else int(len(data) / n_proc))

        arguments = [[func, Q, part, i, use_worker_id]
                     for i, part in enumerate([data[i: i + step] for i in range(0, len(data), step)])]
        
    processes = []
    for i in range(n_proc):
        p = proc(target=_do_parallel_data_prefetch, args=arguments[i])
        processes += [p]

    # Start processes
    print(f"Start prefetching...")

    start = time.time()
    gather_res = [[] for _ in range(n_proc)]
    try:
        for p in processes:  p.start()

        k = 0
        while k < n_proc:
            res = Q.get()  # Get process result
            if res == "Done":  k += 1
            else:              gather_res[res[0]] = res[1]

    except Exception as e:
        for p in processes:  p.terminate()
        raise e

    finally:
        for p in processes:  p.join()
        print(f"Prefetching complete. [{time.time() - start} sec.]")

    if target_data_type == 'ndarray':
        if not isinstance(gather_res[0], np.ndarray):
            return np.concatenate([np.asarray(r) for r in gather_res], axis=0)

        return np.concatenate(gather_res, axis=0)  # Order outputs
    
    elif target_data_type == 'list':
        out = []
        for r in gather_res:  out.extend(r)
        return out
    
    else:
        return gather_res

###

def imread_uint(path, n_channels=3):  # Get uint8 image of size HxWxn_channles (RGB)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB if img.ndim == 2 else cv2.COLOR_BGR2RGB)   # HxWx3(GGG or RGB)

###

def imsave(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:  img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)

###

def uint2single(img):  # numpy(unit) <--->  numpy(single) [0, 1]
    return np.float32(img/255.)

###

def single2uint(img):  # numpy(single) [0, 1] <--->  numpy(unit)
    return np.uint8((img.clip(0, 1)*255.).round())

###


def cubic(x):  # Matlab 'imresize' function, now only support 'bicubic'
    absx = torch.abs(x);  absx2 = absx**2;  absx3 = absx**3

    return (1.5*absx3 - 2.5*absx2 + 1) * ((absx <= 1).type_as(absx)) + \
           (-0.5*absx3 + 2.5*absx2 - 4*absx + 2) * (((absx > 1)*(absx <= 2)).type_as(absx))

###

def calculate_weights_indices(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    # Use a modified kernel to simultaneously interpolate and antialias- larger kernel width
    if (scale < 1) and (antialiasing):
        kernel_width = kernel_width / scale

    x = torch.linspace(1, out_length, out_length)  # Output-space coordinates

    # Input-space coordinates. Calculate the inverse mapping such that 0.5 in output space maps to 0.5 in input space, 
    # and (0.5 + scale) in output space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = torch.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the computation?
    P = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output pixel are in row k of the indices matrix.
    indices = left.view(out_length, 1).expand(out_length, P) + torch.linspace(0, P - 1, P).view(1, P).expand(out_length, P)

    # The weights used to compute the k-th output pixel are in row k of the weights matrix.
    distance_to_center = u.view(out_length, 1).expand(out_length, P) - indices

    # Apply cubic kernel
    weights = scale * cubic(distance_to_center * scale) if (scale < 1) and (antialiasing) else cubic(distance_to_center)

    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, P)

    # If a column in weights is all zero, get rid of it. only consider the first and last column.
    weights_zero_tmp = torch.sum((weights == 0), 0)

    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, P - 2);  weights = weights.narrow(1, 1, P - 2)

    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, P - 2);  weights = weights.narrow(1, 0, P - 2)

    weights = weights.contiguous()
    indices = indices.contiguous()

    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length

    indices = indices + sym_len_s - 1

    return weights, indices, int(sym_len_s), int(sym_len_e)

###

def imresize_np(img, scale, antialiasing=True):  # imresize for numpy image [0, 1]
    img = torch.from_numpy(img)
    if img.dim() == 2:  img.unsqueeze_(2)

    in_H, in_W, in_C = img.size()
    out_C, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = 'cubic'

    # Return the desired dimension order for performing the resize.  
    # The strategy is to perform the resize first along the dimension with the smallest scale factor.

    # Get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(in_W, out_W, scale, kernel, kernel_width, antialiasing)

    # Process H dimension
    img_aug = torch.FloatTensor(in_H + sym_len_Hs + sym_len_He, in_W, in_C)
    img_aug.narrow(0, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:sym_len_Hs, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, 0, sym_len_Hs).copy_(sym_patch_inv)


    sym_patch = img[-sym_len_He:, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()

    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(out_H, in_W, in_C)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        for j in range(out_C):
            out_1[i, :, j] = img_aug[idx:idx + kernel_width, :, j].transpose(0, 1).mv(weights_H[i])

    # Process W dimension
    out_1_aug = torch.FloatTensor(out_H, in_W + sym_len_Ws + sym_len_We, in_C)
    out_1_aug.narrow(1, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :sym_len_Ws, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()

    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, -sym_len_We:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(out_H, out_W, in_C)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        for j in range(out_C):
            out_2[:, i, j] = out_1_aug[:, idx:idx + kernel_width, j].mv(weights_W[i])
    
    if img.dim() == 2:  out_2.squeeze_()

    return out_2.numpy()