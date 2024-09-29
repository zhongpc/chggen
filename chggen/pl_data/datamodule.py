import random
from typing import Optional
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

# from chggen.common.utils import PROJECT_ROOT
from chggen.common.data_utils import get_scaler_from_data_list


def worker_init_fn(id: int) -> None:
    """
    DataLoaders workers init function.

    Initialize the numpy.random seed correctly for each worker, so that
    random augmentations between workers and/or epochs are not identical.

    If a global seed is set, the augmentations are deterministic.

    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)
    return None

class CrystDataModule(pl.LightningDataModule):
    """Crystal data module for packing traning, validation and testing data."""
    def __init__(
        self,
        train_dataset: Dataset = None,
        val_dataset: Dataset = None,
        test_dataset: Dataset = None,
        num_workers: int = 1,
        batch_size: int = 16,
    ) -> None:
        """Initialize the module with the given dataset."""
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.num_workers = num_workers
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        """Download the dataset."""
        pass

    def setup(self, stage: Optional[str] = None):
        """construct datasets and assign data scalers."""
        pass

    def train_dataloader(self) -> DataLoader:
        """Returns a DataLoader for training."""
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns a DataLoader for validation."""
        return DataLoader(
                self.val_dataset,
                shuffle=False,
                batch_size=self.batch_size, # need to improve
                num_workers=self.num_workers,
                worker_init_fn=worker_init_fn,
            )

    def test_dataloader(self) -> DataLoader:
        """Returns a DataLoader for testing."""
        return DataLoader(
                self.test_dataset,
                shuffle=False,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                worker_init_fn=worker_init_fn,
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.train_dataset=}, "
            f"{self.num_workers=}, "
            f"{self.batch_size=})"
        )