import pytorch_lightning as L
from torch.utils.data import DataLoader
import math
import torch


class ImageDataModule(L.LightningDataModule):
    """
    Module to load image data
    """

    def __init__(
        self,
        dataset_builder,
        full_batch_size,
        num_workers,
        num_nodes=1,
        num_devices=1,
    ):
        super().__init__()
        self.batch_size = full_batch_size // (num_nodes * num_devices)
        print(f"Each GPU will receive {self.batch_size} images")
        self.num_workers = num_workers
        self._dataset_builder = dataset_builder

    def setup(self, stage=None):
        self.train_dataset, self.val_dataset = self._dataset_builder()
        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Val dataset size: {len(self.val_dataset)}")
        self.train_aug = self.train_dataset.transform
        self.val_aug = self.val_dataset.transform

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            prefetch_factor=4,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
