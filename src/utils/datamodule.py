import lightning.pytorch as L
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
        full_val_batch_size,
        num_workers,
        num_nodes=1,
        num_devices=1,
    ):
        super().__init__()
        self.batch_size = full_batch_size // (num_nodes * num_devices)
        print(f"Each GPU will receive {self.batch_size} images")
        self.val_batch_size = full_val_batch_size // (num_nodes * num_devices)
        self.num_workers = num_workers
        self._dataset_builder = dataset_builder

    def setup(self, stage=None):
        self.train_dataset, self.val_dataset = self._dataset_builder()
        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Val dataset size: {len(self.val_dataset)}")
        self.train_aug = self.train_dataset.transform
        self.val_aug = self.val_dataset.transform

    def train_dataloader(self):
        shuffle = True
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset):
            shuffle = False
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            prefetch_factor=8,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            # shuffle=True,
            num_workers=1,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
        )


class DataModule(L.LightningDataModule):
    """
    Module to load image data
    """

    def __init__(
        self,
        dataset_builder,
        full_batch_size,
        full_val_batch_size,
        num_workers,
        num_nodes=1,
        num_devices=1,
    ):
        super().__init__()
        self.batch_size = full_batch_size // (num_nodes * num_devices)
        print(f"Each GPU will receive {self.batch_size} samples")
        self.val_batch_size = full_val_batch_size // (num_nodes * num_devices)
        self.num_workers = num_workers
        self._dataset_builder = dataset_builder

    def setup(self, stage=None):
        self.train_dataset, self.val_dataset = self._dataset_builder()
        print("Train and val datasets built")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            prefetch_factor=2,
            pin_memory=False,
            persistent_workers=True,
            drop_last=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=1,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
        )
