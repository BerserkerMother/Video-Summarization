import os
import tarfile
import tempfile
import shutil
import pickle

import glob
import numpy as np

import get_annotation
import reduce_fps
import feature_extraction as ft
from .segmentations import get_segment_fn

ACCEPTED_VIDEO_FORMATS = ["mp4", "mkv", "mpeg"]


def create_tvsum_dataset(path: str):
    """
    create tvsum dataset as tar.gz file

    directory looks like below:
        .
    ├── README
    ├── ydata-tvsum50-data
    ├── ydata-tvsum50-matlab
    ├── ydata-tvsum50-thumbnail
    └── ydata-tvsum50-video

    :param path: path to dataset folder
    :type path: str
    """

    # create temp directory to save files
    temp = tempfile.mkdtemp()

    # read annotations
    mat_path = "ydata-tvsum50-matlab/matlab/ydata-tvsum50.mat"
    mat_path = os.path.join(path, mat_path)
    video_path = "ydata-tvsum50-video"
    video_path = os.path.join(path, video_path)
    # annotations is a dictionary where each key is video_id and values are
    # namedtuple(category, gt_score, n_frame, title, user_anno, video_id)
    annotations = get_annotation.get_summe_annotation(mat_path)

    # extract features and add n_step and picks to annotations
    final_annotations = {}
    for extra in process_features(video_path, temp, fps=2):
        video_name, indices, n_steps, segments = extra
        # add number of steps and picks to final annotations
        anno = annotations[video_name]._asdict()
        anno["n_steps"] = n_steps
        anno["picks"] = indices
        anno["change_points"] = segments
        final_annotations[video_name] = anno

    # save annotations as pickle file
    annotation_loc = os.path.join(temp, "annotations")
    with open(annotation_loc, "wb") as file:
        pickle.dump(final_annotations, file)
    make_tar(temp, "haha.tar.gz")

    # shutil.rmtree(temp)


def create_summe_dataset(path: str, fps: int = 2):
    """
    create summe dataset as tar.gz file

    directory looks like below:
    .
    ├── demo.m
    ├── GT
    ├── matlab
    ├── python
    ├── README.txt
    └── videos

    :param path: path to SumMe directory
    :type path: str
    :param fps: frame per second of output file
    :type fps: int
    """

    # create temp directory to save files
    temp = tempfile.mkdtemp()
    # read annotations and videos
    video_files = "videos"
    video_path = os.path.join(path, video_files)
    mat_path = "GT"
    mat_path = os.path.join(path, mat_path)
    # annotations is a dictionary where each key is video_id and values are
    # namedtuple(category, gt_score, n_frame, title, user_anno, video_id) change this~~~~~~~~~~~~~~~~~~~~~~~~
    annotations = get_annotation.get_summe_annotation(mat_path)

    # extract features and add n_step and picks to annotations
    final_annotations = {}
    for extra in process_features(video_path, temp, fps=2):
        video_name, indices, n_steps, segments = extra
        # add number of steps and picks to final annotations
        anno = annotations[video_name]._asdict()
        anno["n_steps"] = n_steps
        anno["picks"] = indices
        anno["change_points"] = segments
        final_annotations[video_name] = anno

    # save annotations as pickle file
    annotation_loc = os.path.join(temp, "annotations")
    with open(annotation_loc, "wb") as file:
        pickle.dump(final_annotations, file)
    make_tar(temp, "haha.tar.gz")

    # shutil.rmtree(temp)


def make_tar(path, tar_name):
    """
    compress files in path with tar

    :param path: directory to compress
    :type path: str
    :param tar_name: tar file name
    :type tar_name: str
    """
    with tarfile.open(tar_name, "w:gz") as tar_handle:
        for root, dirs, files in os.walk(path):
            for file in files:
                tar_handle.add(os.path.join(root, file))


def process_features(video_path, temp, fps: int = 2):
    """
    Saves frame and video features to temp directory

    :param video_path: path to video folder
    :type video_path: str
    :param temp: folder to save features to
    :type temp: str
    :param fps: down sampled video fps
    :type fps: int
    :returns: iterator over all videos giving (video name, indices, n_steps)
    """
    # create directory to save features in
    ft_dir = os.path.join(temp, "features")
    frame_dir = os.path.join(ft_dir, "frames")
    video_dir = os.path.join(ft_dir, "video")
    os.mkdir(ft_dir)
    os.mkdir(frame_dir)
    os.mkdir(video_dir)

    video_paths = glob.glob(
        video_path + "/**/*",
        recursive=True)

    for path in video_paths:
        file_format = path.split(".")[-1]
        if file_format in ACCEPTED_VIDEO_FORMATS:
            format_length = len(file_format) + 1
            video_name = path.split("/")[-1][:-format_length]
            # reduce fps
            video_array, indices, num_frames = reduce_fps.reduce_fps(
                video_path=path, fps=fps)
            n_steps = len(indices)

            # extract video and frame level features
            frame_ft = ft.get_google_net_features(video_array, size=224)
            video_ft = ft.get_video_feature(video_array, size=122)
            # save features
            frame_ft_path = os.path.join(frame_dir, video_name + ".npy")
            video_ft_path = os.path.join(video_dir, video_name + ".npy")
            np.save(frame_ft_path, frame_ft)
            np.save(video_ft_path, video_ft)
            # produce segments
            seg_fn = get_segment_fn(mode="uniform")
            segments = seg_fn(num_frames)
            # yield sampled video annotations
            yield video_name, indices, n_steps, segments

# create_summe_dataset("/home/kave/Downloads/VS datasets/SumMe", fps=2)
# create_tvsum_dataset("/home/kave/Downloads/VS datasets/tvsum50_ver_1_1/ydata-tvsum50-v1_1/")
