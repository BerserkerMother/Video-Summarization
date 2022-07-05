import numpy as np
from typing import Tuple

import cv2 as cv


def reduce_fps(video_path: str, fps: int = 2) -> Tuple:
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

    # dead video file
    cap = cv.VideoCapture(video_path)
    # get video information
    original_num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    original_frame_rate = int(cap.get(cv.CAP_PROP_FPS))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    # number of total frames after down sampling
    final_num_frames = original_num_frames * fps // original_frame_rate
    # step to sample frames, here we sample frames uniformly
    step_size = original_frame_rate // fps

    # create numpy array that will hold video, shape(T, H, W ,C)
    # here T is just the number of frames in down sampled video
    final_video = np.zeros(shape=(final_num_frames, height, width, 3),
                           dtype=np.uint8)
    # loop over video frames
    current_frame_index = 0
    i = 0
    # stores indices of selected frames
    frame_indices = []
    ret = True
    while ret and i != final_num_frames:
        cap.grab()
        if current_frame_index % step_size == 0:
            ret, frame_array = cap.retrieve()
            arr = np.empty_like(frame_array)
            arr[:, :, 0] = frame_array[:, :, 2]
            arr[:, :, 1] = frame_array[:, :, 1]
            arr[:, :, 2] = frame_array[:, :, 0]
            final_video[i] = arr.astype(np.uint8)
            frame_indices.append(current_frame_index)
            i += 1
        current_frame_index += 1
    # stores indices of selected frames
    frame_indices = np.array(frame_indices)
    return final_video, frame_indices, original_num_frames
