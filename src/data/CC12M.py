"""Implements the normal data module."""

import os
import tarfile

import torch
import torch.nn.functional as F
import numpy as np
from glob import glob
import webdataset as wds
from functools import partial
import lightning.pytorch as L
from torch.distributed import get_world_size
from tqdm import tqdm

## Implements the collate function ##
def collate_fn(batch, max_text_embedding_tokens=64, embedding_size=2048):
    """Implements a collate function for the dataloader.

    :param batch: A batch of data. It consists the following elements:
        - images: The images or the img latents.
        - conditions: The condition for the images.
        - masks: The masks for the img latents.
    :param max_text_embedding_tokens: The maximum number of text embedding tokens.
    :param embedding_size: The size of the embedding.
    """

    ## Unpack the batch ##
    img_latents, text_embeddings, masks, txt = zip(*batch)

    ## Extracting the batch size ##
    batch_size = len(text_embeddings)

    ## Batch and return the data ##
    batch = {}
    batch["img_latents"] = torch.as_tensor(np.stack(img_latents), dtype=torch.float).contiguous()
    # batch["text_embeddings"] = torch.nested.nested_tensor(
    #     list(text_embeddings), dtype=torch.float
    # ).to_padded_tensor(
    #     padding=0.0, output_size=(batch_size, max_text_embedding_tokens, embedding_size)
    # ).contiguous()
    batch["text_embeddings"] = torch.stack(text_embeddings).contiguous()
    batch["masks"] = torch.stack(masks).contiguous()
    batch["txt"] = txt

    return batch


## Imagenet Latent Data Module ##
class CC12MDataModule(L.LightningDataModule):
    """Implements the latent data module for the imagenet dataset."""

    def __init__(
        self,
        root_dir: str,
        max_text_embedding_tokens=64,
        embedding_size=2048,
        batch_size: int = 32,
        val_batch_size: int = 32,
        num_workers: int = 8,
        img_size: int = 256,
    ):
        """Constructor.

        :param root_dir: str: Directory where the data is stored.
        :param max_text_embedding_tokens: int: Maximum number of tokens in the text embedding.
        :param batch_size: int: Batch size for the data loader.
        :param num_workers: int: Number of workers for the data loader.
        :param img_size: int: Image size.
        """
        super().__init__()

        self.root_dir = root_dir


        self.max_text_embedding_tokens = max_text_embedding_tokens
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.tar_keys = (
            [
                f"vae_embeddings_{img_size}.npy",
                "flan_t5_xl_embeddings.npy",
                "txt",
            ]
        )

    def _transform(self, data, train=True):
        """Transform the data necessarily.

        :param data: tuple: The data from the tar file.
        :param train: bool: Whether the data is training data or not.
        """

        ## Retrieving the data and making the img latent ##
        img_latent, text_embedding, txt = data

        ## Truncating the condition upto max token ##
        condition = text_embedding[: self.max_text_embedding_tokens]

        ## Retrieving the token length of the condition ##
        text_embedding_length = len(condition)

        # pad condition
        condition = F.pad(torch.as_tensor(condition, dtype=torch.float), (0, 0, 0, self.max_text_embedding_tokens-text_embedding_length), "constant", 0)

        ## Making the mask ##
        mask = torch.arange(self.max_text_embedding_tokens) < text_embedding_length

        return (
            img_latent,
            condition,
            mask,
            txt
        )

    def _make_dataset(self, train: bool = True):
        """Creates a webdataset dataset."""
        ## Fetching the urls ##
        if train:
            dir = "train" if os.path.isdir(f"{self.root_dir}/train") else "train_wds"
            files = sorted(glob(f"{self.root_dir}/{dir}/*.tar"))
        else:
            dir = "val"
            files = sorted(glob(f"{self.root_dir}/{dir}/*.tar"))

        # count = 0
        # for f in tqdm(files):
        #     with tarfile.open(f, 'r') as tar:
        #         # count += len(tar.getmembers())
        #         for m in tar:
        #             count +=1
        # if train:
        #     self.train_dataset_size = count//6
        # else:
        #     self.val_dataset_size = count//6

        first_tar_name = os.path.basename(files[0]).split(".")[0]
        last_tar_name = os.path.basename(files[-1]).split(".")[0]

        urls = f"{self.root_dir}/{dir}/{{{first_tar_name}..{last_tar_name}}}.tar"

        ## Creating the dataset ##
        dataset = (
            wds.WebDataset(
                urls,
                shardshuffle=train,  ## Shuffle the shards for workers
                nodesplitter=wds.split_by_node,
                workersplitter=wds.split_by_worker,
            )
            .shuffle(
                1000 if train else False,
            )  ## Create a buffer of 1k samples from the workers and shuffle them
            .decode()  ## Decode the data
            .to_tuple(*self.tar_keys)  ## Converting to tuple
            .map(partial(self._transform, train=train))  ## Transforming the data
            .batched(
                self.batch_size if train else self.val_batch_size,
                collation_fn=partial(
                    collate_fn,
                    max_text_embedding_tokens=self.max_text_embedding_tokens,
                    embedding_size=self.embedding_size,
                ),
                partial=False,
            )  ## Already batch samples from the tar files
        )

        return dataset

    def _make_loader(self, train: bool = True):
        """Creates a webdataset loader."""
        batch_size = self.batch_size if train else self.val_batch_size
        ## Creating the dataloader ##
        loader = (
            wds.WebLoader(
                self.train_dataset if train else self.val_dataset,
                num_workers=self.num_workers if train else 1,
                batch_size=None,
            )
            # .with_length(
            #     (self.train_dataset_size if train else self.val_dataset_size)
            #     // (batch_size * get_world_size())
            # )
            # .with_epoch(
            #     (self.train_dataset_size if train else self.val_dataset_size)
            #     // (batch_size * get_world_size())
            # )
        )

        return loader

    def setup(self, stage=None):
        """Setup the data."""

        ## Creating the datasets ##
        self.train_dataset = self._make_dataset(train=True)
        self.val_dataset = self._make_dataset(train=False)

    def train_dataloader(self):
        """Returns the train data loader."""
        return self._make_loader(train=True)

    def val_dataloader(self):
        """Returns the val data loader."""
        return self._make_loader(train=False)