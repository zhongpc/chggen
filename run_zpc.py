from pathlib import Path
from datetime import datetime
from typing import List

import numpy as np
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything, Callback
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

from chggen.pl_data.datamodule import CrystDataModule
from chggen.pl_data.dataset import CHGNetDataset, process_csv


from chggen.pl_modules.encoder import CHGNet_encoder
from chggen.pl_modules.model import CHGGen
# process_csv(input_file='/home/zhongpc/cdvae/data/perov_5/test_zpc.csv',
#             num_workers = 8, 
#             niggli  = True, 
#             primitive = False, 
#             prop_list = ['heat_ref'])


# dataset = CHGNetDataset(path= '/home/zhongpc/chggen/data/perov_5/test_zpc.csv',
#                         name = 'zpc_test',
#                         prop_list = ['heat_ref', 'heat_all'],
#                         )

# item = dataset[0]

encoder = CHGNet_encoder().load()

chggent = CHGGen()


print("Done")