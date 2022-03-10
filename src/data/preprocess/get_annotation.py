import os
from collections import namedtuple

import h5py
import numpy as np
import glob
from scipy import io


def get_tv_annotation(path: str):
    """
    return tvsum annotations
    :param path: path to mat file
    :type path: str
    :returns: dictionary (video_id, data)
    """

    # create named tuple to store values
    Data = namedtuple("Data", "category gt_score n_frame "
                              "title user_anno video_id")
    # dictionary to store data
    dataset = {}
    # read matlab file
    with h5py.File(path, 'r') as file:
        annotations = file["tvsum50"]
        categories = annotations["category"]
        gt_scores = annotations["gt_score"]
        n_frames = annotations["nframes"]
        titles = annotations["title"]
        user_annos = annotations["user_anno"]
        video = annotations["video"]

        num_videos = categories.shape[0]
        for i in range(num_videos):
            # get the category, damn h5py refs
            cat = categories[i][0]
            cat = file[cat]
            cat = "".join(chr(char) for char in cat)

            # read gt score
            gt_score = gt_scores[i][0]
            gt_score = file[gt_score]
            gt_score = np.array(gt_score).reshape(-1)
            # scores are between [0, 4], normalize them
            # gt_score = gt_score / 4

            # number of frames
            n_frame = n_frames[i][0]
            n_frame = file[n_frame]
            n_frame = np.array(n_frame, dtype=np.int64).reshape(-1)

            # video title
            title = titles[i][0]
            title = file[title]
            title = "".join(chr(char) for char in title)

            # user annotations
            user_anno = user_annos[i][0]
            user_anno = file[user_anno]
            user_anno = np.array(user_anno, dtype=np.uint8)

            # video
            video_id = video[i][0]
            video_id = file[video_id]
            video_id = "".join(chr(char) for char in video_id)

            data = Data(cat, gt_score, n_frame, title, user_anno, video_id)
            dataset[video_id] = data
    return dataset


def get_summe_annotation(path: str):
    """
    return summe annotations
    :param path: path to directory containing mat files
    :type path: str
    :returns: dictionary (video_id, data)
    """

    # create named tuple to store values
    Data = namedtuple("Data", "gt_score n_frame "
                              "title user_anno segment")
    # to store data
    dataset = {}
    # get each video annotation path
    path = os.path.join(path, "*.mat")
    file_paths = glob.glob(path)
    for p in file_paths:
        file = io.loadmat(p)
        segment = file["segments"]
        n_frame = file["nFrames"].reshape(-1)
        gt_score = file["gt_score"].reshape(-1)
        user_score = file["user_score"].transpose(1, 0)
        name = p.split("/")[-1].split(".")[0]
        data = Data(gt_score, n_frame, name, user_score, segment)
        dataset[name] = data
    return dataset
