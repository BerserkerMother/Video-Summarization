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
        self.user_summaries = []
        with h5py.File(os.path.join(root, PATH[dataset]), 'r') as hdf:
            for key in hdf.keys():
                self.data.append(hdf[key]['features'][...].astype(np.float32))
                self.target.append(hdf[key]['gtscore'][...].astype(np.float32))

                video_index = key[6:]
                user_summary = np.array(
                    hdf.get('video_' + video_index + '/user_summary'))
                user_scores = hdf[key]["user_scores"][...]
                sb = np.array(hdf.get('video_' + video_index + '/change_points'))
                n_frames = np.array(hdf.get('video_' + video_index + '/n_frames'))
                positions = np.array(hdf.get('video_' + video_index + '/picks'))
                self.user_summaries.append(
                    UserSummaries(user_summary, user_scores, key,
                                  sb, n_frames, positions))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.data[idx])
        targets = torch.tensor(self.target[idx])
        user_summary = self.user_summaries[idx]
        return features, targets, user_summary


def collate_fn(batch):
    features, targets, user_summaries = zip(*batch)
    features = torch.stack(features, dim=0)
    targets = torch.stack(targets, dim=0)
    return features, targets, user_summaries


class UserSummaries:
    def __init__(self, user_summary, user_scores, name,
                 changes_point, n_frames, picks):
        self.user_summary = user_summary
        self.user_scores = user_scores
        self.change_points = changes_point
        self.n_frames = n_frames
        self.picks = picks
        self.name = name
