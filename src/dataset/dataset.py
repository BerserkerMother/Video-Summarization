import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
from typing import List
import os
import numpy as np
import h5py

from .path import PATH


# Dataset Implementation for DS-net TVsum & SumMe

class TSDataset(Dataset):
    def __init__(self, root, dataset, key):
        self.root = root
        self.key = key
        self.files_name = self.get_datasets(self.key)

        self.data = []
        self.target = []
        self.name = []
        with h5py.File(os.path.join(root, PATH[dataset]), 'r') as f:
            for key in self.files_name:
                self.data.append(f[key]['features'][...].astype(np.float32))
                self.target.append(f[key]['gtscore'][...].astype(np.float32))
                self.name.append(key)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.data[idx])
        targets = torch.tensor(self.target[idx])
        vid_name = self.name[idx]

        return features, targets, vid_name

    def get_datasets(self, keys: List[str]):
        files_name = [str(Path(key).name) for key in keys]
        # datasets = [h5py.File(path, 'r') for path in dataset_paths]
        return files_name


class PreTrainDataset(Dataset):
    def __init__(self, root):
        self.root = root

        self.data = []
        self.target = []
        self.name = []
        for dataset in PATH.keys():
            with h5py.File(os.path.join(root, PATH[dataset]), 'r') as f:
                for key in f.keys():
                    self.data.append(f[key]['features']
                                     [...].astype(np.float32))
                    self.name.append(key)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.data[idx])
        vid_name = self.name[idx]

        return features, vid_name
