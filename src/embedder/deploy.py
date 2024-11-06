import re
import time
import cv2
import torch
import numpy as np
import argparse, os, sys, glob

from PIL import Image
from tqdm import trange
from omegaconf import OmegaConf
from torch.utils.data.dataloader import default_collate

from main import instantiate_from_config
from generator.src.modules.distributions import DiagonalGaussianDistribution

###

def save_image(x, path, dtype="uint8"):
    if dtype == 'uint8':    Image.fromarray((torch.clamp(x.detach().cpu(), 0, 1).numpy().transpose(1, 2, 0) * 255).astype(np.uint8)).save(path)
    elif dtype == 'uint16': 
        img = (torch.clamp(x.detach().cpu(), 0, 1).numpy().transpose(1, 2, 0) * 2**16).astype(np.uint16)
        # cv2.imwrite(path, img)
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

###

def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-r", "--resume", type=str, nargs="?", 
                        help="load from logdir or checkpoint in logdir",)
    
    parser.add_argument("-b", "--base", nargs="*", metavar="base_config.yaml", default=list(),
                        help="paths to base configs. Loaded from left-to-right. "
                             "Parameters can be overwritten or added with command-line options of the form `--key value`.",)
    
    parser.add_argument("-c", "--config", nargs="?", metavar="single_config.yaml", const=True, default="",
                        help="path to single config. If specified, base configs will be ignored "
                             "(except for the last one if left unspecified).",)
    
    parser.add_argument("--ignore_base_data", action="store_true",
                        help="Ignore data specification from base configs. Useful if you want "
                             "to specify a custom datasets on the command line.",)
    
    parser.add_argument("--outdir", required=False, type=str,
                        help="Where to write outputs to.",)
    
    parser.add_argument("--top_k", type=int, default=100,
                        help="Sample from among top-k predictions.",)
    
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature.",)
    
    return parser

###

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

###

@torch.no_grad()
def load_model_and_dataset(config, ckpt, gpu, eval_mode):

    dataset_array = instantiate_from_config(config.data)
    dataset_array.prepare_data()
    dataset_array.setup()

    if ckpt:
        pl_state_dict = torch.load(ckpt, map_location="cpu")
        global_step = pl_state_dict["global_step"]
        
        # *******************************************************************************************************
        if False:  # Experiment: RAW-RAW Encoder and RAW-RGB Decoder (Purpose of the decoder)
            state_dict = pl_state_dict["state_dict"]

            # NOTE: Removing old embedder keys from encoder model
            # https://gist.github.com/lucascoelhof/158980602ddfc90b0da52b10d3ce06b7
            pattern = re.compile('encoder.*')
            matched_keys = {key for key, _ in state_dict.items() if pattern.match(key)}
            for key in matched_keys:  state_dict.pop(key, None)

            encoder_ckpt = '/home/ubuntu/scania-raw-diff/src/embedder/logs/checkpoints/vqgan/f4_d3/raw_raw_r256_e10.ckpt'
            
            state_dict_encoder = torch.load(encoder_ckpt, map_location = 'cpu')['state_dict']

            # NOTE: Updating encoder-embedder keys
            pattern = re.compile('encoder.*')
            matched_keys = {key for key, _ in state_dict_encoder.items() if pattern.match(key)}
            for key in matched_keys:  state_dict.update({key : state_dict_encoder[key]})

            pl_state_dict["state_dict"] = state_dict

        # *******************************************************************************************************
    else:
        pl_state_dict = {"state_dict": None}
        global_step = None

    model = load_model_from_config(config.model, pl_state_dict["state_dict"], gpu=gpu, eval_mode=eval_mode)["model"]

    return dataset_array, model, global_step

###

@torch.no_grad()
def run_conditional(model, dataset_array, outdir, top_k, temperature, batch_size=1):

    if len(dataset_array.datasets) > 1:
        split = sorted(dataset_array.datasets.keys())[0]
        dataset = dataset_array.datasets[split]
    else:
        dataset = next(iter(dataset_array.datasets.values()))
    print("Dataset: ", dataset.__class__.__name__)

    for start_idx in trange(0, len(dataset) - batch_size + 1, batch_size):

        x_indices = list(range(start_idx, start_idx + batch_size))
        x_collate = default_collate([dataset[i] for i in x_indices])

        x_array = model.get_input(x_collate, "image").to(model.device)

        # VQGAN Model
        # xz_array, _, _ = model.encode(x_array)
        # x_hat = model.decode(xz_array)

        # VQVAE Model
        xz_array = model.encode(x_array)

        # *******************************************************************************************************
        if False:   # Experiment: Latent interpolation at different ratios
            xz_array = model.quant_conv(model.encoder(x_array))

            for idx, ratio in enumerate(range(1, 11, 1)):
                xz_distr = DiagonalGaussianDistribution((xz_array[1] * (1-(ratio/10)) + xz_array[2] * (ratio/10)).unsqueeze(0))
                x_hat = model.decode(xz_distr.sample())
                save_image(x_hat[0], os.path.join(outdir, "{:06}.png".format(idx)))

            break
        # *******************************************************************************************************

        x_hat = model.decode(xz_array.sample())

        for i in range(x_hat.shape[0]):
            save_image(x_hat[i], os.path.join(outdir, x_collate["path"][0]))

###

if __name__ == "__main__":

    ckpt = None
    show_config = False
    gpu, eval_mode = True, True
    
    sys.path.append(os.getcwd())
    option, unknown = get_parser().parse_known_args()

    if option.resume:  # Resume execution from checkpoint
        
        if not os.path.exists(option.resume):  
            raise ValueError("Cannot find {}".format(option.resume))

        paths = option.resume.split("/")

        try:                idx = len(paths) - paths[::-1].index("logs") + 1
        except ValueError:  idx = -2

        logdir = "/".join(paths[:idx])
        ckpt = option.resume
            
        print(f"logdir:{logdir}")
        base_config_array = sorted(glob.glob(os.path.join(logdir, "configs/*-project.yaml")))
        option.base = base_config_array + option.base

    if option.config:  # Set configuration if defined or select default
        option.base = [option.config] if type(option.config) == str else [option.base[-1]]

    config_array = [OmegaConf.load(cfg) for cfg in option.base]

    if option.ignore_base_data:  # Ignore default data
        for config in config_array:
            if hasattr(config, "data"): del config["data"]

    config = OmegaConf.merge(*config_array, OmegaConf.from_dotlist(unknown))

    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False

    dataset_array, model, global_step = load_model_and_dataset(config, ckpt, gpu, eval_mode)
    print(f"Global step: {global_step}")

    os.makedirs(option.outdir, exist_ok=True)
    run_conditional(model, dataset_array, option.outdir, option.top_k, option.temperature)
