import numpy as np


def uniform_segmentation(n_frames: int, sec_per_seg: int = 2, fps: int = 2):
    """
    segments video uniformly

    :param n_frames: total number of frames in down sampled video
    :type n_frames: int
    :param sec_per_seg: total duration of each segment as second
    :type sec_per_seg: int
    :param fps: down sampled video fps
    :type fps: int
    :returns: np array containing index of frames
    """

    frame_per_seg = fps * sec_per_seg
    segments = np.arange(start=0, stop=n_frames, step=frame_per_seg)
    return segments


if __name__ == '__main__':
    s = uniform_segmentation(200)
    print(s)
