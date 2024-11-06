import re, time
import numpy as np
import torch, torchvision

from PIL import Image
from tqdm import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from embedder.deploy import get_parser

from generator.src.modules.diffusion.ddim import DDIMSampler
from generator.src.utils import ismap, instantiate_from_config, default

###

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")

    model = instantiate_from_config(config.model)
    state_dict = pl_sd["state_dict"]

    # https://gist.github.com/lucascoelhof/158980602ddfc90b0da52b10d3ce06b7
    # pattern = re.compile('first_stage_model.*')
    # matched_keys = {key for key, _ in state_dict.items() if pattern.match(key)}
    # for key in matched_keys:  state_dict.pop(key, None)

    model.load_state_dict(state_dict, strict=False);  model.cuda();  model.eval()

    return {"model": model}, pl_sd["global_step"]

@torch.no_grad()
def convsample_ddim(model, cond, steps, shape, eta=1.0, callback=None, normals_sequence=None, mask=None, x0=None, quantize_x0=False, img_callback=None,
                    temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None, x_T=None, log_every_t=None):

    # print(f"Sampling with eta = {eta}; steps: {steps}")
    return DDIMSampler(model).sample(steps, shape[0], shape[1:], cond, callback, normals_sequence, img_callback, quantize_x0, eta, mask, x0, temperature, 
                                     noise_dropout, score_corrector, corrector_kwargs, False, x_T)

###

@torch.no_grad()
def make_convolutional_sample(batch, model, mode="vanilla", custom_steps=None, eta=1.0, swap_mode=False, masked=False, invert_mask=True, quantize_x0=False, 
                              custom_schedule=None, decode_interval=1000, resize_enabled=False, custom_shape=None, temperature=1., noise_dropout=0., 
                              corrector=None, corrector_kwargs=None, x_T=None, save_intermediate_vid=False, make_progrow=True, ddim_use_x0_pred=False):

    z, c, x, x_target, xrec, xc = model.get_input(batch, model.first_stage_key, return_first_stage_outputs=True, return_original_cond=True,
                                        force_c_encode=not (hasattr(model, 'split_input_params') and model.cond_stage_key == 'coordinates_bbox'))

    log_every_t = 1 if save_intermediate_vid else None

    if custom_shape is not None:
        z = torch.randn(custom_shape)
        print(f"Generating {custom_shape[0]} samples of shape {custom_shape[1:]}")

    log = dict();  log["input"] = x;  log["reconstruction"] = xrec; log["target"] = c

    if ismap(xc):
        log["original_conditioning"] = model.to_rgb(xc)
        if hasattr(model, 'cond_stage_key'):
            log[model.cond_stage_key] = model.to_rgb(xc)

    else:
        log["original_conditioning"] = xc if xc is not None else torch.zeros_like(x)
        if model.cond_stage_model:
            log[model.cond_stage_key] = xc if xc is not None else torch.zeros_like(x)
            if model.cond_stage_key =='class_label':
                log[model.cond_stage_key] = xc[model.cond_stage_key]

    with model.ema_scope("Plotting"):
        t0 = time.time()
        sample, intermediates = convsample_ddim(model, c, steps=custom_steps, shape=z.shape, eta=eta, quantize_x0=quantize_x0, img_callback=None, mask=None, x0=None,
                                                temperature=temperature, noise_dropout=noise_dropout, score_corrector=corrector, corrector_kwargs=corrector_kwargs,
                                                x_T=x_T, log_every_t=log_every_t)
        t1 = time.time()

        if ddim_use_x0_pred:  sample = intermediates['pred_x0'][-1]

    x_sample = model.decode_first_stage(sample)

    try:
        x_sample_noquant = model.decode_first_stage(sample, force_not_quantize=True)
        log["sample_noquant"] = x_sample_noquant
        log["sample_diff"] = torch.abs(x_sample_noquant - x_sample)
    except:  pass
    
    log["time"] = t1 - t0
    log["intermediates"] = intermediates
    log["sample"] = x_sample

    return log

###

@torch.no_grad()
def run(model, img, custom_steps, resize_enabled=False, classifier_ckpt=None, global_step=None):
    x_T = None
    # x_T = model.first_stage_model.encode(rearrange(img['image'], 'b h w c -> b c h w').to(torch.device('cuda'))).sample()
    # x_T = model.q_sample(x_start=x_T, 
    #                      t=torch.randint(0, model.num_timesteps, (x_T.shape[0],), device=torch.device('cuda')).long(),
    #                      noise=torch.randn_like(x_T))
    
    height, width = img["image"].shape[1:3]
    split_input = False # height >= 128 and width >= 128

    if split_input:
        ks = 128;  stride = 64;  vqf = 4
        model.split_input_params = {"ks": (ks, ks), 
                                    "stride": (stride, stride),
                                    "vqf": vqf,
                                    "patch_distributed_vq": True,
                                    "tie_braker": False,
                                    "clip_max_weight": 0.5,
                                    "clip_min_weight": 0.01,
                                    "clip_max_tie_weight": 0.5,
                                    "clip_min_tie_weight": 0.01}
    else:
        if hasattr(model, "split_input_params"):  delattr(model, "split_input_params")

    return make_convolutional_sample(img, model, mode='ddim', custom_steps=custom_steps, eta=1., swap_mode=False , masked=False, invert_mask=False, 
                                     quantize_x0=False, custom_schedule=None, decode_interval=10, resize_enabled=resize_enabled, custom_shape=None,
                                     temperature=1., noise_dropout=0., corrector=None, corrector_kwargs=None, x_T=x_T, save_intermediate_vid=False,
                                     make_progrow=True,ddim_use_x0_pred=False)

###

if __name__ == "__main__":

    # Load configuration
    option, unknown = get_parser().parse_known_args()
    config = OmegaConf.merge(*[OmegaConf.load(cfg) for cfg in [option.config]], OmegaConf.from_dotlist(unknown))
    
    # Instantiate and load data
    x = instantiate_from_config(config.data);  x.prepare_data();  x.setup()

    model, _ = load_model_from_config(config, option.resume)

    for idx, x_elem in tqdm(enumerate(x.val_dataloader())):

        output = model["model"].decode_first_stage(run(model["model"], x_elem, custom_steps = 10)["intermediates"]["pred_x0"][-1])
        # output = run(model["model"], x_elem, custom_steps = 5)["sample"]
        sample = torch.clamp(output.detach().cpu() , -1., 1.)
        sample = (sample + 1.) * 127.5  # Transformations of data to avoid artifacts in output image
        Image.fromarray(np.transpose((sample).numpy().astype(np.uint8), (0, 2, 3, 1))[0]).save(option.outdir + '/' + x_elem["path"][0])