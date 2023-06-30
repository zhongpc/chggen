from pathlib import Path
from datetime import datetime
from typing import List
from types import SimpleNamespace
import pickle
import numpy as np
import torch
# import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
# from pytorch_lightning import seed_everything, Callback
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

from chggen.pl_data.datamodule import CrystDataModule
from chggen.pl_data.dataset import CHGNetDataset, process_csv


from chggen.pl_modules.encoder import CHGNet_encoder
from chggen.pl_modules.decoder import GemNetTDecoder
from chggen.pl_modules.model import CHGGen

from chggen.common.data_utils import get_scaler_from_data_list

device = torch.device("cuda:7")

with open('./output_from_ld', 'rb') as fp:
    output = pickle.load(fp)

print("Done")
