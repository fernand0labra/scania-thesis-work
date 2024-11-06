import torch
import signal
import pytorch_lightning as pl
import argparse, os, sys, datetime, importlib

from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from torch.utils.data import DataLoader, Dataset

###

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:  importlib.reload(importlib.import_module(module))
    return getattr(importlib.import_module(module, package=None), cls)

###

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):                             return v
        if v.lower() in ("yes", "true", "t", "y", "1"):     return True
        elif v.lower() in ("no", "false", "f", "n", "0"):   return False
        else:                                               raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    
    parser.add_argument("-n", "--name", type=str, const=True, default="", nargs="?",
        help="model name",)

    parser.add_argument("-r", "--resume", type=str, const=True, default="", nargs="?",
                        help="resume from logdir or checkpoint in logdir",)
    
    parser.add_argument("-b", "--base", nargs="*", metavar="base_config.yaml", default=list(),
                        help="paths to base configs. Loaded from left-to-right. ""Parameters can be overwritten or added with command-line options of the form `--key value`.",)
    
    parser.add_argument("-t", "--train", type=str2bool, const=True, default=False, nargs="?",
                        help="train",)
    
    parser.add_argument("--no-test", type=str2bool, const=True, default=False, nargs="?",
                        help="disable test",)

    parser.add_argument("-s", "--seed", type=int, default=23,
                        help="seed for seed_everything",)

    return parser

###

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

###

def melk(*args, **kwargs):  # Allow checkpointing via USR1
    if trainer.global_rank == 0:
        trainer.save_checkpoint(os.path.join(ckptdir, "last.ckpt"))

###

class WrappedDataset(Dataset):  # Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

###

class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, wrap=False, num_workers=None):
        super().__init__()

        self.wrap = wrap
        self.batch_size = batch_size
        self.dataset_configs = dict()

        self.num_workers = num_workers if num_workers is not None else batch_size * 2

        if train is not None:       self.dataset_configs["train"] = train;              self.train_dataloader = self._train_dataloader
        if validation is not None:  self.dataset_configs["validation"] = validation;    self.val_dataloader = self._val_dataloader
        if test is not None:        self.dataset_configs["test"] = test;                self.test_dataloader = self._test_dataloader


    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)


    def setup(self, stage=None):
        self.datasets = dict((k, instantiate_from_config(self.dataset_configs[k])) for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])


    def _train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False) # , collate_fn=custom_collate


    def _val_dataloader(self):
        return DataLoader(self.datasets["validation"], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False) # , collate_fn=custom_collate


    def _test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False) # , collate_fn=custom_collate

###

# Custom parser to specify config files, train, test and debug mode, postfix, resume.
#       "--key value"       arguments are interpreted as arguments to the trainer.
#       "nested.key=value"  arguments are interpreted as config parameters.

if __name__ == "__main__":

    sys.path.append(os.getcwd())
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    option, unknown = get_parser().parse_known_args()
    
    if option.resume:
        if not os.path.exists(option.resume):  raise ValueError("Cannot find {}".format(option.resume))

        if os.path.isfile(option.resume):
            paths = option.resume.split("/")
            idx = len(paths) - paths[::-1].index("logs")
            logdir = "/".join(paths[:idx])
            ckpt = option.resume

        else:
            assert os.path.isdir(option.resume), option.resume
            logdir = option.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        _tmp = logdir.split("/")
        nowname = _tmp[_tmp.index("logs")]

    else:
        if option.name:    name = "_" + option.name
        elif option.base:  name = "_" + os.path.splitext(os.path.split(option.base[0])[-1])[0]
        else:           name = ""

        nowname = now + name # + option.postfix
        logdir = os.path.join("/home/ubuntu/scania-raw-diff/src/embedder/logs/optim/uncond/logs", nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(option.seed)

    # Init and save configs
    config = OmegaConf.merge(*[OmegaConf.load(cfg) for cfg in option.base], OmegaConf.from_dotlist(unknown))
    lightning_config = config.pop("lightning", OmegaConf.create())
    
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    for k in range(0, len(unknown), 2):  # Skip '--'in the flags
        try:     trainer_config[unknown[k][2:]] = int(unknown[k+1])
        except:  trainer_config[unknown[k][2:]] = unknown[k+1]  
    
    # https://github.com/CompVis/taming-transformers/issues/59
    model = instantiate_from_config(config.model)
    state_dict = torch.load(ckpt, map_location = 'cpu')['state_dict'] # .state_dict()
    model.load_state_dict(state_dict, strict = False)

    lightning_config.trainer = trainer_config
    trainer_kwargs = dict()  # Trainer and callbacks

    default_logger_cfgs = {
        "target": "pytorch_lightning.loggers.TensorBoardLogger",
        "params": {
            "name": nowname,
            "save_dir": os.path.join('/',*logdir.split('/')[:-1]),
        }
    }

    trainer_kwargs["logger"] = instantiate_from_config(default_logger_cfgs)

    # Add callback which sets up log directory
    default_callbacks_cfg = {
        "setup_callback": {
            "target": "pytorch_lightning.callbacks.SetupCallback",
            "params": {
                "resume": option.resume,
                "now": now,
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                "config": config,
                "lightning_config": lightning_config,
            }
        },
        "model_callback":{
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
            }
        }
    }
    
    if hasattr(model, "monitor"):
        print(f"Monitoring {model.monitor} as checkpoint metric.")
        default_callbacks_cfg["model_callback"]["params"]["monitor"] = model.monitor

    trainer_kwargs["callbacks"] = [instantiate_from_config(default_callbacks_cfg[k]) for k in default_callbacks_cfg]
    
    trainer = Trainer(**trainer_config, **trainer_kwargs, strategy="ddp_find_unused_parameters_true", num_nodes=1) # strategy="ddp_find_unused_parameters_true", num_nodes=1
    data = instantiate_from_config(config.data);  data.prepare_data();  data.setup()

    # Configure learning rate
    bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate

    model.learning_rate = bs * base_lr
    print("Setting learning rate to {:.2e} = {} (batchsize) * {:.2e} (base_lr)".format(model.learning_rate, bs, base_lr))

    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False

    signal.signal(signal.SIGUSR1, melk)
    if option.train:
        try:                trainer.fit(model, data)
        except Exception:   melk(); raise

    if not option.no_test and not trainer.interrupted:
        trainer.test(model, data)