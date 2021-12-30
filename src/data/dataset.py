import numpy as np
import h5py
import os

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from .path import PATH


# Dataset Implementation for DS-net TVsum & SumMe

class TSDataset(Dataset):
    def __init__(self, root, dataset='tvsum'):
        self.root = root
        self.dataset = dataset
        self.data = []
        self.target = []
        self.name = []
        with h5py.File(os.path.join(root, PATH[dataset]), 'r') as file:
            for key in file.keys():
                self.data.append(file[key]['features'][...].astype(np.float32))
                self.target.append(file[key]['gtscore'][...].astype(np.float32))
                self.name.append(key)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.data[idx])
        targets = torch.tensor(self.target[idx])
        vid_name = self.name[idx]
        return features, targets, vid_name


def collate_fn(batch):
    features, targets, name = zip(*batch)
    features = pad_sequence(features, batch_first=True)
    targets = pad_sequence(targets, batch_first=True)
    return features, targets, name
