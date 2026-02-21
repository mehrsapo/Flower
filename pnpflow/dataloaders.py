import torch
import torchvision
import torchvision.transforms as v2
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from PIL import Image
import pandas as pd
import os
import warnings
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pickle
import logging


class DataLoaders:
    def __init__(self, dataset_name, batch_size_train, batch_size_test):
        self.dataset_name = dataset_name
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test

    def load_data(self):

        if self.dataset_name == 'celeba':
            transform = v2.Compose([
                v2.CenterCrop(178),
                v2.Resize((128, 128)),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            # Paths
            img_dir = './data/celeba/img_align_celeba/img_align_celeba'
            partition_csv = './data/celeba/list_eval_partition.csv'

            # Datasets
            train_dataset = CelebADataset(
                img_dir, partition_csv, partition=0, transform=transform)
            val_dataset = CelebADataset(
                img_dir, partition_csv, partition=1, transform=transform)
            test_dataset = CelebADataset(
                img_dir, partition_csv, partition=2, transform=transform)

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size_train,
                shuffle=True,
                collate_fn=custom_collate)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size_test,
                shuffle=False,
                collate_fn=custom_collate)
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size_test,
                shuffle=False,
                collate_fn=custom_collate)

        elif self.dataset_name == 'celebahq':

            transform = v2.Compose([
                v2.Resize(256),
                v2.ToTensor(),         # Convert images to PyTorch tensor
            ])

            test_dir = './data/celebahq/test/'
            test_dataset = CelebAHQDataset(
                test_dir, batchsize=self.batch_size_test, transform=transform)
            train_loader = None
            val_loader = None
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size_test,
                shuffle=False,
                collate_fn=custom_collate)

        elif self.dataset_name == 'afhq_cat':
            # transform should include a linear transform 2x - 1
            transform = v2.Compose([
                v2.Resize((256, 256)),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

            # transform = False
            img_dir_test = './data/afhq_cat/test/cat/'
            img_dir_val = './data/afhq_cat/val/cat/'
            img_dir_train = './data/afhq_cat/train/cat/'
            test_dataset = AFHQDataset(
                img_dir_test, batchsize=self.batch_size_test, transform=transform, split='test')
            val_dataset = AFHQDataset(
                img_dir_val, batchsize=self.batch_size_test, transform=transform, split='val')
            train_dataset = AFHQDataset(
                img_dir_train, batchsize=self.batch_size_test, transform=transform, split='train')
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size_test,
                shuffle=False,
                collate_fn=custom_collate)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size_test,
                shuffle=False,
                collate_fn=custom_collate)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size_train,
                shuffle=True,
                collate_fn=custom_collate, drop_last=True)

        else:
            raise ValueError("The dataset your entered does not exist")

        data_loaders = {'train': train_loader,
                        'test': test_loader, 'val': val_loader}

        return data_loaders


class CelebADataset(Dataset):
    def __init__(self, img_dir, partition_csv, partition, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.partition = partition

        # Load the partition file correctly
        partition_df = pd.read_csv(
            partition_csv, header=0, names=[
                'image', 'partition'], skiprows=1)
        self.img_names = partition_df[partition_df['partition']
                                      == partition]['image'].values

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)

        if not os.path.exists(img_path):
            warnings.warn(f"File not found: {img_path}. Skipping.")
            return None, None

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, 0


class CelebAHQDataset(Dataset):
    """CelebA HQ dataset."""

    def __init__(self, data_dir, batchsize, transform=None):
        self.files = os.listdir(data_dir)
        self.root_dir = data_dir
        self.num_imgs = len(os.listdir(self.root_dir))
        self.transform = transform
        self.batchsize = batchsize

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        img_name = self.files[idx]
        img_path = os.path.join(self.root_dir, img_name)

        if not os.path.exists(img_path):
            warnings.warn(f"File not found: {img_path}. Skipping.")
            return None, None

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
            image = 2 * image - 1
        image = image.float()

        return image, 0


import os
import warnings
from PIL import Image
from torch.utils.data import Dataset

class AFHQDataset(Dataset):
    """AFHQ Cat dataset.
       - train/val: keep old behavior (scan img_dir)
       - test: read file names from a list file
    """

    def __init__(self, img_dir, batchsize, category='cat', transform=None,
                 split='train'):
        self.img_dir   = img_dir
        self.batchsize = batchsize
        self.category  = category
        self.transform = transform
        self.split     = split

        list_file=os.path.join(os.path.dirname(__file__), '..', 'data_splits', '400_filenames.txt')
        if split in ('train', 'val'):
            # Old behavior
            self.files = sorted(os.listdir(self.img_dir))
        elif split == 'test':
            if list_file is None:
                raise ValueError("For split='test', you must provide list_file=path/to/filenames.txt")

            # Load names from list_file (ignore blanks and comments)
            with open(list_file, "r", encoding="utf-8") as f:
                wanted = [
                    line.strip().rstrip("\r")
                    for line in f
                    if line.strip() and not line.lstrip().startswith("#")
                ]

            # Resolve to full paths; if the entry has a path separator, treat as relative to img_dir;
            # absolute paths are used as-is.
            resolved, missing = [], []
            for name in wanted:
                if os.path.isabs(name):
                    path = name
                elif os.sep in name:
                    path = os.path.normpath(os.path.join(self.img_dir, name))
                else:
                    path = os.path.join(self.img_dir, name)

                if os.path.exists(path):
                    resolved.append(path)
                else:
                    missing.append(path)

            if missing:
                preview = "\n  ".join(missing[:5])
                warnings.warn(
                    f"{len(missing)} files listed in {list_file} were not found under {self.img_dir}.\n"
                    f"First few missing:\n  {preview}"
                )

            if not resolved:
                raise RuntimeError("No valid images found from the provided list_file for split='test'.")

            # For test we store full paths to avoid extra joins later
            self.files = resolved
        else:
            raise ValueError("split must be one of: 'train', 'val', 'test'")

        self.num_imgs = len(self.files)

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        entry = self.files[idx]

        # train/val entries are basenames; test entries are full paths
        if self.split in ('train', 'val'):
            img_path = os.path.join(self.img_dir, entry)
        else:  # test
            img_path = entry

        # Safety: skip missing files at access time (should be rare after init)
        if not os.path.exists(img_path):
            warnings.warn(f"File not found at access time: {img_path}. Skipping.")
            return None, None

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0



def custom_collate(batch):
    # Filter out None values

    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) == 0:
        return torch.tensor([]), torch.tensor([])
    return torch.utils.data._utils.collate.default_collate(batch)


logging.basicConfig(level=logging.INFO)