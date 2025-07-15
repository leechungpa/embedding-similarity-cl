import os

import omegaconf
from omegaconf import OmegaConf
from solo.utils.auto_resumer import AutoResumer
from solo.utils.checkpointer import Checkpointer
from solo.utils.misc import omegaconf_select

try:
    from solo.data.dali_dataloader import PretrainDALIDataModule
except ImportError:
    _dali_available = False
else:
    _dali_available = True

try:
    from solo.utils.auto_umap import AutoUMAP
except ImportError:
    _umap_available = False
else:
    _umap_available = True

_N_CLASSES_PER_DATASET = {
    "cifar10": 10,
    "cifar100": 100,
    "cifar100coarse": 20,
    "stl10": 10,
    "imagenet": 1000,
    "imagenet100": 100,
}

_SUPPORTED_DATASETS = [
    "cifar10",
    "cifar100",
    "cifar100coarse",
    "stl10",
    "imagenet",
    "imagenet100",
    "custom",
]


def add_and_assert_dataset_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
    """Adds specific default values/checks for dataset config.

    Args:
        cfg (omegaconf.DictConfig): DictConfig object.

    Returns:
        omegaconf.DictConfig: same as the argument, used to avoid errors.
    """

    assert not OmegaConf.is_missing(cfg, "data.dataset")
    assert not OmegaConf.is_missing(cfg, "data.train_path")

    assert cfg.data.dataset in _SUPPORTED_DATASETS

    # if validation path is not available, assume that we want to skip eval
    cfg.data.val_path = omegaconf_select(cfg, "data.val_path", None)
    cfg.data.format = omegaconf_select(cfg, "data.format", "image_folder")
    cfg.data.no_labels = omegaconf_select(cfg, "data.no_labels", False)
    cfg.data.fraction = omegaconf_select(cfg, "data.fraction", -1)
    cfg.debug_augmentations = omegaconf_select(cfg, "debug_augmentations", False)

    return cfg


def add_and_assert_wandb_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
    """Adds specific default values/checks for wandb config.

    Args:
        cfg (omegaconf.DictConfig): DictConfig object.

    Returns:
        omegaconf.DictConfig: same as the argument, used to avoid errors.
    """

    cfg.wandb = omegaconf_select(cfg, "wandb", {})
    cfg.wandb.enabled = omegaconf_select(cfg, "wandb.enabled", False)
    cfg.wandb.entity = omegaconf_select(cfg, "wandb.entity", None)
    cfg.wandb.project = omegaconf_select(cfg, "wandb.project", "solo-learn")
    cfg.wandb.offline = omegaconf_select(cfg, "wandb.offline", False)

    return cfg


def add_and_assert_lightning_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
    """Adds specific default values/checks for Pytorch Lightning config.

    Args:
        cfg (omegaconf.DictConfig): DictConfig object.

    Returns:
        omegaconf.DictConfig: same as the argument, used to avoid errors.
    """

    cfg.seed = omegaconf_select(cfg, "seed", 5)
    cfg.resume_from_checkpoint = omegaconf_select(cfg, "resume_from_checkpoint", None)
    cfg.strategy = omegaconf_select(cfg, "strategy", None)

    return cfg


def parse_cfg(cfg: omegaconf.DictConfig):
    # default values for checkpointer
    cfg = Checkpointer.add_and_assert_specific_cfg(cfg)

    # default values for auto_resume
    cfg = AutoResumer.add_and_assert_specific_cfg(cfg)

    # default values for dali
    if _dali_available:
        cfg = PretrainDALIDataModule.add_and_assert_specific_cfg(cfg)

    # default values for auto_umap
    if _umap_available:
        cfg = AutoUMAP.add_and_assert_specific_cfg(cfg)

    # assert dataset parameters
    cfg = add_and_assert_dataset_cfg(cfg)

    # default values for wandb
    cfg = add_and_assert_wandb_cfg(cfg)

    # default values for pytorch lightning stuff
    cfg = add_and_assert_lightning_cfg(cfg)

    # extra processing
    if cfg.data.dataset in _N_CLASSES_PER_DATASET:
        cfg.data.num_classes = _N_CLASSES_PER_DATASET[cfg.data.dataset]
    else:
        # hack to maintain the current pipeline
        # even if the custom dataset doesn't have any labels
        cfg.data.num_classes = max(
            1,
            sum(entry.is_dir() for entry in os.scandir(cfg.data.train_path)),
        )

    # find number of big/small crops
    big_size = cfg.augmentations[0].crop_size
    num_large_crops = num_small_crops = 0
    for pipeline in cfg.augmentations:
        if big_size == pipeline.crop_size:
            num_large_crops += pipeline.num_crops
        else:
            num_small_crops += pipeline.num_crops
    cfg.data.num_large_crops = num_large_crops
    cfg.data.num_small_crops = num_small_crops

    if cfg.data.format == "dali":
        assert cfg.data.dataset in ["imagenet100", "imagenet", "custom"]

    # adjust lr according to batch size
    cfg.num_nodes = omegaconf_select(cfg, "num_nodes", 1)
    cfg.optimizer.lr_method = omegaconf_select(cfg, "optimizer.lr_method", "linear")
    if cfg.optimizer.lr_method == "linear":
        # Linear scaling LR 
        scale_factor = cfg.optimizer.batch_size*len(cfg.devices)*cfg.num_nodes / 256
    elif cfg.optimizer.lr_method == "square_root":
        # Square root LR
        scale_factor = (cfg.optimizer.batch_size * len(cfg.devices) * cfg.num_nodes)**(1/2)
    elif cfg.optimizer.lr_method == "without_scaling":
        scale_factor = 1
    else:
        raise ValueError("Not Implemented.")

    cfg.optimizer.lr = cfg.optimizer.lr * scale_factor

    if cfg.data.val_path is not None:
        assert not OmegaConf.is_missing(cfg, "optimizer.classifier_lr")
        cfg.optimizer.classifier_lr = cfg.optimizer.classifier_lr * scale_factor

    # extra optimizer kwargs
    cfg.optimizer.kwargs = omegaconf_select(cfg, "optimizer.kwargs", {})
    if cfg.optimizer.name == "sgd":
        cfg.optimizer.kwargs.momentum = omegaconf_select(cfg, "optimizer.kwargs.momentum", 0.9)
    elif cfg.optimizer.name == "lars":
        cfg.optimizer.kwargs.momentum = omegaconf_select(cfg, "optimizer.kwargs.momentum", 0.9)
        cfg.optimizer.kwargs.eta = omegaconf_select(cfg, "optimizer.kwargs.eta", 1e-3)
        cfg.optimizer.kwargs.clip_lr = omegaconf_select(cfg, "optimizer.kwargs.clip_lr", False)
        cfg.optimizer.kwargs.exclude_bias_n_norm = omegaconf_select(
            cfg,
            "optimizer.kwargs.exclude_bias_n_norm",
            False,
        )
    elif cfg.optimizer.name == "adamw":
        cfg.optimizer.kwargs.betas = omegaconf_select(cfg, "optimizer.kwargs.betas", [0.9, 0.999])

    # modify the job name
    if omegaconf_select(cfg, "name_kwargs.add_method", default=False):
        cfg.name = cfg.name + f"_{cfg.method}"
    if omegaconf_select(cfg, "name_kwargs.add_batch_size", default=False):
        cfg.name = cfg.name + f"_bsz{cfg.optimizer.batch_size}"
    if omegaconf_select(cfg, "name_kwargs.add_weight", default=False):
        cfg.name = cfg.name + f"_w{cfg.add_vrn_loss_term.weight}"
    if cfg.method == "simclr":
        if omegaconf_select(cfg, "name_kwargs.add_temperature", default=False):
            cfg.name = cfg.name + f"_t{cfg.method_kwargs.temperature}"
    if cfg.method == "dhel":
        if omegaconf_select(cfg, "name_kwargs.add_temperature", default=False):
            cfg.name = cfg.name + f"_t{cfg.method_kwargs.temperature}"
    if cfg.method == "dcl":
        if omegaconf_select(cfg, "name_kwargs.add_temperature", default=False):
            cfg.name = cfg.name + f"_t{cfg.method_kwargs.temperature}"
    if cfg.method == "simsiam":
        if omegaconf_select(cfg, "name_kwargs.add_pred_hidden_dim", default=False):
            cfg.name = cfg.name + f"_pred{cfg.method_kwargs.pred_hidden_dim}"
    if cfg.method == "vicreg":
        if omegaconf_select(cfg, "name_kwargs.add_sim_loss_weight", default=False):
            cfg.name = cfg.name + f"_sim{cfg.method_kwargs.sim_loss_weight}"
        if omegaconf_select(cfg, "name_kwargs.add_var_loss_weight", default=False):
            cfg.name = cfg.name + f"_var{cfg.method_kwargs.var_loss_weight}"
        if omegaconf_select(cfg, "name_kwargs.add_cov_loss_weight", default=False):
            cfg.name = cfg.name + f"_cov{cfg.method_kwargs.cov_loss_weight}"
    if cfg.method == "sigmoid":
        if omegaconf_select(cfg, "name_kwargs.add_temperature", default=False):
            cfg.name = cfg.name + f"_t{cfg.method_kwargs.temperature}"
        if omegaconf_select(cfg, "name_kwargs.add_bias", default=False):
            cfg.name = cfg.name + f"_b{cfg.method_kwargs.bias}"

    return cfg
