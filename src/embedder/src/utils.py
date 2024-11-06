import os
import torch
import collections
import numpy as np
import albumentations

from PIL import Image
from omegaconf import OmegaConf

###

def custom_collate(batch):
    elem = batch[0];  elem_type = type(elem)

    if isinstance(elem, torch.Tensor):
        # If we're in a background process, concatenate directly into a shared memory tensor to avoid an extra copy
        return torch.stack(batch, 0, out=None) if torch.utils.data.get_worker_info() is None else \
               torch.stack(batch, 0, out=elem.new(elem.untyped_storage()._new_shared(sum([x.numel() for x in batch]))).reshape((1, 1088, 1088, 3)))
    
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' and elem_type.__name__ != 'string_':
        
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':           return custom_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():                                                          return torch.as_tensor(batch)
        
    elif isinstance(elem, float):                                                       return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):                                                         return torch.tensor(batch)
    elif isinstance(elem, collections.abc.Mapping):                                     return {key: custom_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, collections.abc.Sequence):                                    return [custom_collate(samples) for samples in zip(*batch)]

###

def get_preprocessor(size=None, random_crop=False, additional_targets=None, crop_size=None):
    if size is not None and size > 0:

        transforms = list()
        transforms.append(albumentations.SmallestMaxSize(max_size = size))

        if not random_crop:     transforms.append(albumentations.CenterCrop(height=size,width=size))
        else:                   transforms.append(albumentations.RandomCrop(height=size,width=size)); \
                                transforms.append(albumentations.HorizontalFlip())
        return albumentations.Compose(transforms, additional_targets=additional_targets)

###
    
def imscale(x, factor, keepshapes=False, keepmode="bicubic"):

    h, w, _ = x.shape;  nh = h//factor;  nw = w//factor
    keepmode = {"nearest": Image.NEAREST, "bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC}[keepmode]

    img = (x + 1.0) * 127.5
    img = Image.fromarray(img.clip(0, 255).astype(np.uint8))

    img = img.resize((nw,nh), Image.BICUBIC)
    img = np.array(img)/127.5 - 1.0

    return img.astype(x.dtype)

###

class KeyNotFoundError(Exception):
    def __init__(self, cause, keys=None, visited=None):
        
        self.cause = cause;  self.keys = keys;  self.visited = visited
        messages = list()
        
        if keys is not None:        messages.append("Key not found: {}".format(keys))
        if visited is not None:     messages.append("Visited: {}".format(visited))

        messages.append("Cause:\n{}".format(cause))
        super().__init__("\n".join(messages))

###

def get_ckpt_path(name, root, check=False):
    return os.path.join(root, "vgg.pth")

###

def retrieve(list_or_dict, key, splitval="/", default=None, expand=True, pass_success=False):
    """Given a nested list or dict return the desired value at key expanding
    callable nodes if necessary and :attr:`expand` is ``True``. The expansion
    is done in-place.

    Parameters
    ----------
        list_or_dict : list or dict
            Possibly nested list or dictionary.
        key : str
            key/to/value, path like string describing all keys necessary to
            consider to get to the desired value. List indices can also be
            passed here.
        splitval : str
            String that defines the delimiter between keys of the
            different depth levels in `key`.
        default : obj
            Value returned if :attr:`key` is not found.
        expand : bool
            Whether to expand callable nodes on the path or not.

    Returns
    -------
        The desired value or if :attr:`default` is not ``None`` and the
        :attr:`key` is not found returns ``default``.

    Raises
    ------
        Exception if ``key`` not in ``list_or_dict`` and :attr:`default` is
        ``None``.
    """

    keys = key.split(splitval)
    success = True

    try:
        visited = [];  parent = None;  last_key = None
        for key in keys:

            if callable(list_or_dict):
                if not expand:  raise KeyNotFoundError(ValueError("Trying to get past callable node with expand=False."), keys, visited,)
                list_or_dict = list_or_dict()
                parent[last_key] = list_or_dict

            last_key = key
            parent = list_or_dict

            try:  list_or_dict = list_or_dict[key if isinstance(list_or_dict, dict) else int(key)]
            except (KeyError, IndexError, ValueError) as e:  raise KeyNotFoundError(e, keys=keys, visited=visited)

            visited += [key]

        # final expansion of retrieved value
        if expand and callable(list_or_dict):
            list_or_dict = list_or_dict()
            parent[last_key] = list_or_dict

    except KeyNotFoundError as e:
        if default is None:  raise e
        else:  list_or_dict = default

    if not pass_success:  return list_or_dict
    else:  return list_or_dict, success

###

if __name__ == "__main__":
    retrieve(OmegaConf.create({"keyA": "a", "keyB": "b", "keyC": {"cc1": 1, "cc2": 2,}}), "keyA")

