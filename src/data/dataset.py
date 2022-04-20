import glob

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import os
import numpy as np
import h5py

from .path import PATH


class PreTrainDatasetReady(Dataset):
    def __init__(self, root, datasets):
        self.root = root

        self.data = []
        self.datasets = datasets.split("+")
        for dataset in self.datasets:
            dataset_name = PATH[dataset]
            video_rep_path = os.path.join(root, "video", dataset)
            with h5py.File(os.path.join(root, dataset_name), 'r') as f:
                for key in f.keys():
                    data = torch.tensor(f[key]['features'][...].astype(np.float32))
                    video_rep = np.load("%s/%s.npy" % (video_rep_path, key))
                    self.data.append((data, video_rep))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features, video_rep = self.data[idx]
        video_rep = torch.tensor(video_rep)

        return features, video_rep


class PreTrainDataset(Dataset):
    def __init__(self, root):
        self.root = root

        self.data = []
        frame_paths = os.path.join(root, "frames")
        for frame_path in glob.glob(frame_paths + "/*"):
            video_name = os.path.basename(frame_path).split(".")[0]
            data = np.load(frame_path)
            video_rep = np.load("%s/%s.npy" %
                                (os.path.join(root, "video"), video_name))
            self.data.append((data, video_rep))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features, video_rep = self.data[idx]
        features = torch.tensor(features)
        video_rep = torch.tensor(video_rep)

        return features, video_rep


def collate_fn_pretrain(batch):
    features, vid_reps = zip(*batch)
    features = pad_sequence(features, batch_first=True, padding_value=1000)
    vid_reps = torch.stack(vid_reps, dim=0)
    return features, vid_reps
