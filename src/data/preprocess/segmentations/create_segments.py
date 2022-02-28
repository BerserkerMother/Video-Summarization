import numpy as np

from .kts import kts_segmentation
from .uniform import uniform_segmentation


def get_segment_fn(mode: str = "uniform"):
    """
    returns a segmentation function

    :param mode: segmentation method, choice:[uniform, kts]
    :type mode: str
    :returns: segmentation function
    """
    if mode == "uniform":
        return uniform_seg
    elif mode == "kts":
        return kts_seg
    else:
        raise NotImplementedError


def kts_seg(features: np.array, num_seg: int,
            v_max: float, kernel: str = "dot"):
    """
    segments video using kernel temporal segmentation
    https://hal.inria.fr/hal-01022967/PDF/video_summarization.pdf

    :param features: (n, feature_dim) array containing frame features
    :type features: numpy array
    :param num_seg: number of segments to divide to
    :type num_seg: int
    :param v_max: special parameter
    :type v_max: float
    :param kernel: kernel function to use
    :type kernel: str
    :returns: numpy array of frame indices
    """

    if kernel == "dot":
        similarities = np.dot(features, features.T)
        segments, costs = kts_segmentation(similarities,
                                           num_seg,
                                           v_max)
    else:
        raise NotImplementedError
    return segments


def uniform_seg(n_frames: int, sec_per_seg: int = 2, fps: int = 2):
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
    return uniform_segmentation(n_frames, sec_per_seg, fps)
