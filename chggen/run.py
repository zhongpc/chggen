from pathlib import Path
from datetime import datetime
from typing import List

import hydra
import numpy as np
import torch
import omegaconf
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything, Callback
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

from chggen.common.utils import log_hyperparameters, PROJECT_ROOT


def build_callbacks(cfg: DictConfig) -> List[Callback]:
    callbacks: List[Callback] = []

    if "lr_monitor" in cfg.logging:
        hydra.utils.log.info("Adding callback <LearningRateMonitor>")
        callbacks.append(
            LearningRateMonitor(
                logging_interval=cfg.logging.lr_monitor.logging_interval,
                log_momentum=cfg.logging.lr_monitor.log_momentum,
            )
        )

    if "early_stopping" in cfg.train:
        hydra.utils.log.info("Adding callback <EarlyStopping>")
        callbacks.append(
            EarlyStopping(
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                patience=cfg.train.early_stopping.patience,
                verbose=cfg.train.early_stopping.verbose,
            )
        )

    if "model_checkpoints" in cfg.train:
        hydra.utils.log.info("Adding callback <ModelCheckpoint>")
        callbacks.append(
            ModelCheckpoint(
                dirpath=Path('/home/zhongpc/test/wabdb'),
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                save_top_k=cfg.train.model_checkpoints.save_top_k,
                verbose=cfg.train.model_checkpoints.verbose,
            )
        )

    return callbacks


def run(cfg: DictConfig) -> None:
    """
    Generic train loop

    :param cfg: run configuration, defined by Hydra in /conf
    """
    if cfg.train.deterministic:
        seed_everything(cfg.train.random_seed)

    if cfg.train.pl_trainer.fast_dev_run:
        hydra.utils.log.info(
            f"Debug mode <{cfg.train.pl_trainer.fast_dev_run=}>. "
            f"Forcing debugger friendly configuration!"
        )
        # Debuggers don't like GPUs nor multiprocessing
        cfg.train.pl_trainer.gpus = 0
        cfg.data.datamodule.num_workers.train = 0
        cfg.data.datamodule.num_workers.val = 0
        cfg.data.datamodule.num_workers.test = 0

        # Switch wandb mode to offline to prevent online logging
        cfg.logging.wandb.mode = "offline"

    # Hydra run directory
    hydra_dir = Path('/home/zhongpc/test/hydra') # Path(HydraConfig.get().run.dir)

    # Instantiate datamodule
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )

    # Instantiate model
    hydra.utils.log.info(f"Instantiating <{cfg.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )

    # Pass scaler from datamodule to model
    hydra.utils.log.info(f"Passing scaler from datamodule to model <{datamodule.scaler}>")
    model.lattice_scaler = datamodule.lattice_scaler.copy()
    model.scaler = datamodule.scaler.copy()
    torch.save(datamodule.lattice_scaler, hydra_dir / 'lattice_scaler.pt')
    torch.save(datamodule.scaler, hydra_dir / 'prop_scaler.pt')
    # Instantiate the callbacks
    callbacks: List[Callback] = build_callbacks(cfg=cfg)

    # Logger instantiation/configuration
    wandb_logger = None
    if "wandb" in cfg.logging:
        hydra.utils.log.info("Instantiating <WandbLogger>")
        wandb_config = cfg.logging.wandb
        
        wandb_logger = WandbLogger(
            **wandb_config,
            tags=cfg.core.tags,
        )
        hydra.utils.log.info("W&B is now watching <{cfg.logging.wandb_watch.log}>!")
        wandb_logger.watch(
            model,
            log=cfg.logging.wandb_watch.log,
            log_freq=cfg.logging.wandb_watch.log_freq,
        )

    # Store the YaML config separately into the wandb dir
    yaml_conf: str = OmegaConf.to_yaml(cfg=cfg)
    (hydra_dir / "hparams.yaml").write_text(yaml_conf)

    # Load checkpoint (if exist)
    ckpts = list(hydra_dir.glob('*.ckpt'))
    if len(ckpts) > 0:
        ckpt_epochs = np.array([int(ckpt.parts[-1].split('-')[0].split('=')[1]) for ckpt in ckpts])
        ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
        hydra.utils.log.info(f"found checkpoint: {ckpt}")
    else:
        ckpt = None
          
    hydra.utils.log.info("Instantiating the Trainer")

    print(cfg)

    trainer = pl.trainer.trainer.Trainer(
        default_root_dir=hydra_dir,
        logger=wandb_logger,
        callbacks=callbacks,
        deterministic=cfg.train.deterministic,
        check_val_every_n_epoch=cfg.logging.val_check_interval,
        #progress_bar_refresh_rate=cfg.logging.progress_bar_refresh_rate,
        #resume_from_checkpoint=ckpt,
        **cfg.train.pl_trainer,
    )
    log_hyperparameters(trainer=trainer, model=model, cfg=cfg)

    hydra.utils.log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    hydra.utils.log.info("Starting testing!")
    trainer.test(datamodule=datamodule)

    # Logger closing to release resources/avoid multi-run conflicts
    if wandb_logger is not None:
        wandb_logger.experiment.finish()


if __name__ == "__main__":

    now = datetime.now()
    time_tag = now.strftime('%Y-%m-%d')
    PROJECT_ROOT="/home/zhongpc/cdvae"
    HYDRA_JOBS="/home/zhongpc/test/hydra"
    WABDB_DIR="/home/zhongpc/test/wabdb"


    
    cfg_dict = {'data': {'root_path': '${PROJECT_ROOT}/data/perov_5', 'prop': 'heat_ref', 'num_targets': 1, 'niggli': True, 
                         'primitive': False, 'graph_method': 'crystalnn', 'lattice_scale_method': 'scale_length', 'preprocess_workers': 30, 
                         'readout': 'mean', 'max_atoms': 20, 'otf_graph': False, 'eval_model_name': 'perovskite', 'train_max_epochs': 3000, 
                         'early_stopping_patience': 100000, 'teacher_forcing_max_epoch': 1500, 
                         'datamodule': {'_target_': 'cdvae.pl_data.datamodule.CrystDataModule', 
                                        'datasets': {'train': {'_target_': 'cdvae.pl_data.dataset.CrystDataset', 'name': 'Formation energy train', 'path': '${data.root_path}/train.csv', 'prop': '${data.prop}', 'niggli': '${data.niggli}', 'primitive': '${data.primitive}', 'graph_method': '${data.graph_method}', 'lattice_scale_method': '${data.lattice_scale_method}', 'preprocess_workers': '${data.preprocess_workers}'}, 
                                                     'val': [{'_target_': 'cdvae.pl_data.dataset.CrystDataset', 'name': 'Formation energy val', 'path': '${data.root_path}/val.csv', 'prop': '${data.prop}', 'niggli': '${data.niggli}', 'primitive': '${data.primitive}', 'graph_method': '${data.graph_method}', 'lattice_scale_method': '${data.lattice_scale_method}', 'preprocess_workers': '${data.preprocess_workers}'}], 
                                                     'test': [{'_target_': 'cdvae.pl_data.dataset.CrystDataset', 'name': 'Formation energy test', 'path': '${data.root_path}/test.csv', 'prop': '${data.prop}', 'niggli': '${data.niggli}', 'primitive': '${data.primitive}', 'graph_method': '${data.graph_method}', 'lattice_scale_method': '${data.lattice_scale_method}', 'preprocess_workers': '${data.preprocess_workers}'}]}, 
                                        'num_workers': {'train': 0, 'val': 0, 'test': 0}, 
                                        'batch_size': {'train': 512, 'val': 256, 'test': 256}}
                                        }, 
           'logging': {'val_check_interval': 5, 'progress_bar_refresh_rate': 20, 
                       'wandb': {'name': '${expname}', 'project': 'crystal_generation_mit', 'entity': None, 'log_model': True, 'mode': 'online', 'group': '${expname}'}, 
                       'wandb_watch': {'log': 'all', 'log_freq': 500}, 
                       'lr_monitor': {'logging_interval': 'step', 'log_momentum': False}}, 
           'model': {'encoder': {'_target_': 'cdvae.pl_modules.gnn.DimeNetPlusPlusWrap', 'num_targets': '${data.num_targets}', 'hidden_channels': 128, 'num_blocks': 4, 'int_emb_size': 64, 'basis_emb_size': 8, 'out_emb_channels': 256, 'num_spherical': 7, 'num_radial': 6, 'otf_graph': '${data.otf_graph}', 'cutoff': 7.0, 'max_num_neighbors': 20, 'envelope_exponent': 5, 'num_before_skip': 1, 'num_after_skip': 2, 'num_output_layers': 3, 'readout': '${data.readout}'}, 'decoder': {'_target_': 'cdvae.pl_modules.decoder.GemNetTDecoder', 'hidden_dim': 128, 'latent_dim': '${model.latent_dim}', 'max_neighbors': '${model.max_neighbors}', 'radius': '${model.radius}', 'scale_file': '${oc.env:PROJECT_ROOT}/cdvae/pl_modules/gemnet/gemnet-dT.json'}, '_target_': 'cdvae.pl_modules.model.CDVAE', 'hidden_dim': 256, 'latent_dim': 256, 'fc_num_layers': 1, 'max_atoms': '${data.max_atoms}', 'cost_natom': 1.0, 'cost_coord': 10.0, 'cost_type': 1.0, 'cost_lattice': 10.0, 'cost_composition': 1.0, 'cost_edge': 10.0, 'cost_property': 1.0, 'beta': 0.01, 'teacher_forcing_lattice': True, 'teacher_forcing_max_epoch': '${data.teacher_forcing_max_epoch}', 'max_neighbors': 20, 'radius': 7.0, 'sigma_begin': 10.0, 'sigma_end': 0.01, 'type_sigma_begin': 5.0, 'type_sigma_end': 0.01, 'num_noise_level': 50, 'predict_property': False}, 
           'optim': {'optimizer': {'_target_': 'torch.optim.Adam', 'lr': 0.001, 'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0}, 'use_lr_scheduler': True, 'lr_scheduler': {'_target_': 'torch.optim.lr_scheduler.ReduceLROnPlateau', 'factor': 0.6, 'patience': 30, 'min_lr': 0.0001}}, 
           'train': {'deterministic': False, 'random_seed': 42, 
                     'pl_trainer': {'fast_dev_run': False, 'devices': 'auto', 'precision': 32, 
                                    'max_epochs': 20, 'accumulate_grad_batches': 1, 
                                    'num_sanity_val_steps': 2, 'gradient_clip_val': 0.5, 'gradient_clip_algorithm': 'value', 
                                    'profiler': 'simple'}, 'monitor_metric': 'val_loss', 'monitor_metric_mode': 'min', 
                                    'early_stopping': {'patience': '${data.early_stopping_patience}', 'verbose': False}, 
                                    'model_checkpoints': {'save_top_k': 1, 'verbose': False}}, 
           'expname': 'perov', 'core': {'version': '0.0.1', 'tags': [time_tag]}}

    cfg = OmegaConf.create(cfg_dict)


    run(cfg)
