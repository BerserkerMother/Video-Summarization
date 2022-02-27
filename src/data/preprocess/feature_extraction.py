import numpy as np
from PIL import Image

import torch
from torchvision import transforms

import models


def get_google_net_features(video: np.array, size: int = 224):
    """
    Given a video extracts frame level features

    :param video: numpy array of video (T, H, W, 3)
    :type video: numpy array
    :param size: transforms resize size
    :type size: int or tuple
    :returns: numpy array of extracted frame features (T, 1024)
    :rtype: numpy array
    """

    # get model with pretrain weights
    # make sure to change return feature of google net in pytorch
    # implementation
    model, image_transformation = _get_model_and_transforms(
        model_name="google", size=size)

    # covert frame arrays to PIL images
    images = []
    for i in range(video.shape[0]):
        image = Image.fromarray(video[i])
        images.append(image)
    # create frame tensors
    video = []
    for image in images:
        image_tensor = image_transformation(image)
        video.append(image_tensor)
    video = torch.stack(video, dim=0)

    # ready to forward
    frame_features = model.forward(video)
    return frame_features.numpy()


def get_video_feature(video: np.array, size: int = 112):
    """
    Given a video extracts video level features

    :param video: numpy array of video (T, H, W, 3)
    :type video: numpy array
    :param size: transforms resize size
    :type size: int or tuple
    :returns: numpy array of extracted frame features (video_feature,)
    :rtype: numpy array
    """

    # get model with pretrain weights
    # make sure to change return feature of google net in pytorch
    # implementation
    model, image_transformation = _get_model_and_transforms(
        model_name="r3d18", size=size)

    # covert frame arrays to PIL images
    images = []
    for i in range(video.shape[0]):
        image = Image.fromarray(video[i])
        images.append(image)
    # create frame tensors
    video = []
    for image in images:
        image_tensor = image_transformation(image)
        video.append(image_tensor)
    # model expect video of shape (3, T, H, W)
    video = torch.stack(video, dim=0).permute(1, 0, 2, 3).unsqueeze(0)
    video_feature = model(video).view(-1)
    return video_feature.numpy()


MODELS = {
    "google": models.GoogleNet,
    "r3d18": models.R3D18
}
# normalize values are from
# https://pytorch.org/vision/stable/models.html#video-classification
# https://pytorch.org/hub/pytorch_vision_googlenet/
NORMALIZE_VALUES = {
    "r3d18": ([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989]),
    "google": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
}


def _get_model_and_transforms(model_name: str = "google", size: int = 224):
    """
    outputs a feature extractor and data transformation function

    :param model_name: model name, choice:["google", "r3d18"]
    :type model_name: str
    :param size: transforms resize size
    :type size: int or tuple
    :returns: pytorch model and transform function
    """

    # create model
    model = MODELS[model_name]()

    # create transforms
    mean, std = NORMALIZE_VALUES[model_name]
    image_transformation = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                             std=std)
    ])
    return model, image_transformation
