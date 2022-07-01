"""
This module holds the all the code for generating summary images given the model
"""
import os
import logging
from typing import Dict

import torch
import torch.nn as nn
from torch.utils import data

import numpy as np
import cv2 as cv
import json
import glob
from PIL import Image

from evaluation.generate_summary import generate_summary as gm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_video_frames(video_dataset_path: str):
    """
    generate down sample video frames as png


    :param video_dataset_path: path to the folder containing dataset videos
    """

    logging.info("Generating video frames as png")
    videos_path = glob.glob(
        video_dataset_path + "/**/*",
        recursive=True)
    for path in videos_path:
        reduce_fps_and_save(path, fps=2)


def generate_video_summary_json(model: nn.Module, data_loader: data.DataLoader):
    """
    saves summary frames as json

    :param model: model to gte results from
    :param data_loader: data loader to use
    """
    summaries = get_summary(model, data_loader)
    # save as json
    with open("summary.json", 'w') as file:
        json.dump(summaries, file, indent=8)

    logging.info("generating summaries and saving them as json")


@torch.no_grad()
def get_summary(model: nn.Module, data_loader: data.DataLoader) -> Dict:
    """
    takes a model and dataset, outputs generated summaries

    :param model: model to gte results from
    :param data_loader: data loader to use
    """
    model.eval()
    score_dict, user_dict = {}, {}
    for i, (feature, target, user) in enumerate(data_loader):
        feature = feature.to(device)

        pred, _ = model(feature)
        pred = torch.sigmoid(pred.view(1, -1))

        score_dict[user.name] = pred.squeeze(0).detach().cpu().numpy()
        user_dict[user.name] = user
    summaries = generate_summary(score_dict, user_dict)
    names = [("video_%d" % i) for i in range(len(summaries))]
    result = {}

    for (name, summary) in zip(names, summaries):
        new_summary = [i for i, is_in in enumerate(summary) if is_in == 1]
        result[name] = new_summary

    return result


PATH = {
    'ovp': 'eccv16_dataset_ovp_google_pool5.h5',
    'summe': 'eccv16_dataset_summe_google_pool5.h5',
    'tvsum': 'eccv16_dataset_tvsum_google_pool5.h5',
    'youtube': 'eccv16_dataset_youtube_google_pool5.h5'
}


def generate_summary(predicted_dict, user_dict):
    """
    gives a list containing frame numbers of summary for each sample
    """
    all_scores = []
    keys = list(predicted_dict.keys())

    for video_name in keys:  # for each video inside that json file ...
        scores = predicted_dict[video_name]  # read the importance scores from frames
        all_scores.append(scores)

    all_user_summary, all_user_scores, all_shot_bound, \
    all_nframes, all_positions = [], [], [], [], []
    for key in keys:
        user = user_dict[key]
        user_summary = user.user_summary
        user_scores = user.user_scores
        sb = user.change_points
        n_frames = user.n_frames
        positions = user.picks

        all_user_summary.append(user_summary)
        all_user_scores.append(user_scores)
        all_shot_bound.append(sb)
        all_nframes.append(n_frames)
        all_positions.append(positions)

    all_summaries = gm(
        all_shot_bound, all_scores, all_nframes, all_positions)
    return all_summaries


def reduce_fps_and_save(video_path: str, fps: int = 2):
    """
    takes a video path and down sample it to given fps

    :param video_path: path to video to be down sampled
    :type video_path: str
    :param fps: frame per second of output file
    :type fps: int
    :returns: tuple of numpy array consist of down sampled video frames
    and position of select frames in original video
    :rtype: tuple
    """

    if not os.path.exists('movies'):
        os.mkdir("movies")
    video_name = "movies/" + video_path.split("/")[-1].split(".")[0]
    if not os.path.exists(video_name):
        os.mkdir(video_name)
    # dead video file
    cap = cv.VideoCapture(video_path)
    # get video information
    original_num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    original_frame_rate = int(cap.get(cv.CAP_PROP_FPS))
    # number of total frames after down sampling
    final_num_frames = original_num_frames * fps // original_frame_rate
    # step to sample frames, here we sample frames uniformly
    step_size = original_frame_rate // fps

    # loop over video frames
    current_frame_index = 0
    i = 0
    ret = True
    while ret and i != final_num_frames:
        cap.grab()
        if current_frame_index % step_size == 0:
            ret, frame_array = cap.retrieve()
            arr = np.empty_like(frame_array)
            arr[:, :, 0] = frame_array[:, :, 2]
            arr[:, :, 1] = frame_array[:, :, 1]
            arr[:, :, 2] = frame_array[:, :, 0]
            image = Image.fromarray(arr)
            image.save("%s/%d.jpg" % (video_name, i))
            i += 1
        current_frame_index += 1

# generate_video_frames("/home/kave/Downloads/VS datasets/tvsum50_ver_1_1/ydata-tvsum50-v1_1/ydata-tvsum50-video/video")
