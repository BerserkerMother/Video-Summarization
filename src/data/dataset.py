import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import os
import numpy as np
import h5py

from .path import PATH


class PreTrainDataset(Dataset):
    def __init__(self, root, datasets):
        self.root = root

        self.data = []
        self.datasets = datasets.split("+")
        for dataset in self.datasets:
            with h5py.File(os.path.join(root, PATH[dataset]), 'r') as f:
                for key in f.keys():
                    data = torch.tensor(f[key]['features'][...].astype(np.float32))
                    self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.data[idx]

        return features


def collate_fn_pretrain(batch):
    features = batch
    features = pad_sequence(features, batch_first=True, padding_value=1000)
    return features
