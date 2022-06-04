import glob
from pathlib import Path
from typing import List

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
        feature, vid_rep = self.data[idx]
        feature = torch.tensor(feature)
        vid_rep = torch.tensor(vid_rep)
        return feature, vid_rep


# Dataset Implementation for DS-net TVsum & SumMe
class TSDataset(Dataset):
    def __init__(self, root, ex_dataset, datasets,
                 key=None, split: str = "train"):
        """
        :param root: path to *.h5 data folder
        :param ex_dataset: data to benchmark on
        :param datasets: data to use for training
        :param key: specific splitting
        :param split: train or val split
        """
        self.root = root
        self.key = key
        self.split = split
        self.ex_dataset = ex_dataset
        self.datasets = datasets.split("+")

        self.data = []
        self.target = []
        self.user_summaries = []
        # if it's val split, add user summaries to evaluation
        if split == "val":
            with h5py.File(os.path.join(root, PATH[ex_dataset]), 'r') as f:
                # if split keys are given then it reads them,
                # otherwise it adds the whole experiment data
                if key:
                    files_name = self.get_datasets(self.key)
                else:
                    files_name = f.keys()
                for key in files_name:
                    self.data.append(f[key]['features'][...].astype(np.float32))
                    self.target.append(f[key]['gtscore'][...].astype(np.float32))
                    user_summary = np.array(f[key]['user_summary'])
                    user_scores = np.array(f[key]["user_scores"])
                    sb = np.array(f[key]['change_points'])
                    n_frames = np.array(f[key]['n_frames'])
                    positions = np.array(f[key]['picks'])

                    self.user_summaries.append(
                        UserSummaries(user_summary, user_scores, key,
                                      sb, n_frames, positions))
        else:
            for dataset in self.datasets:
                with h5py.File(os.path.join(root, PATH[dataset]), 'r') as f:
                    # if split keys are given then it reads them,
                    # otherwise it adds the whole experiment data
                    if key and dataset == ex_dataset:
                        files_name = self.get_datasets(self.key)
                    else:
                        files_name = f.keys()
                    for key in files_name:
                        features = f[key]['features'][...].astype(np.float32)
                        target = f[key]['gtscore'][...].astype(np.float32)

                        if features.shape[0] > 50:
                            self.data.append(features)
                            self.target.append(target)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.data[idx]
        targets = torch.tensor(self.target[idx])
        if self.split == "train":
            frame_rep = torch.tensor(features)
            return frame_rep, targets
        features = torch.tensor(features)
        return features, targets, self.user_summaries[idx]

    def get_datasets(self, keys: List[str]):
        files_name = [str(Path(key).name) for key in keys]
        # datasets = [h5py.File(path, 'r') for path in dataset_paths]
        return files_name


def collate_fn_pretrain(batch):
    features, vid_reps = zip(*batch)
    features = pad_sequence(features, batch_first=True, padding_value=1000)
    vid_reps = torch.stack(vid_reps, dim=0)
    return features, vid_reps


class UserSummaries:
    def __init__(self, user_summary, user_scores, name,
                 changes_point, n_frames, picks):
        self.user_summary = user_summary
        self.user_scores = user_scores
        self.change_points = changes_point
        self.n_frames = n_frames
        self.picks = picks
        self.name = name


def collate_fn_train(batch):
    features, targets = zip(*batch)
    features = pad_sequence(features, batch_first=True, padding_value=1000)
    targets = pad_sequence(targets, batch_first=True, padding_value=1000)
    return features, targets


def collate_fn_test(batch):
    features, targets, user_summaries = batch[0]
    features = features.unsqueeze(0)
    targets = targets.unsqueeze(0)
    return features, targets, user_summaries
