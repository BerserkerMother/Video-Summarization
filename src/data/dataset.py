import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
from typing import List

import numpy as np
import h5py


# Dataset Implementation for DS-net TVsum & SumMe

class TSDataset(Dataset):
    def __init__(self, root, key):
        self.root = root
        self.key = key
        self.files_name = self.get_datasets(self.key)

        self.data = []
        self.target = []
        self.name = []
        with h5py.File(self.root, 'r') as f:
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
        # dataset_paths = {str(Path(key).parent) for key in keys}
        # keys = [k.repalce('../', '') for k in keys]
        files_name = [str(Path(key).name) for key in keys]
        # datasets = [h5py.File(path, 'r') for path in dataset_paths]
        return files_name


def collate_fn(batch):
    features, targets, name = zip(*batch)

    features = pad_sequence(features, batch_first=True)
    targets = pad_sequence(targets, batch_first=True)

    return features, targets, name
