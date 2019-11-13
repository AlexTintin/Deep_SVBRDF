from __future__ import division
import math
import random
import os

import numpy as np
import torch

#code taken from
#https://github.com/msraig/DeepInverseRendering
#adapted to  Pytorch

def tensor_norm(tensor):
    t = tensor
    Length = torch.sqrt(torch.sum(t*t, axis=0, keepdims=True))
    return torch.div(t, Length + 1e-12)


def tensor_dot(a, b, ax=0):
    return torch.sum((a* b), axis=ax).unsqueeze(0)


def numpy_norm(arr):
    length = np.sqrt(np.sum(arr * arr, axis=-1, keepdims=True))
    return arr / (length + 1e-12)


def numpy_dot(a, b, ax=-1):
    return np.sum(a * b, axis=ax)[..., np.newaxis]


# image utils.
def preprocess(img):
    # [0,1] => [-1,1]
    return img * 2.0 - 1.0


def deprocess(img):
    # [-1,1] => [0,1]
    return (img + 1.0) / 2.0


def toLDR(img):
    return img ** (1.0 / 2.2)


def log_tensor_norm(tensor):
    return (torch.log(torch.add(tensor, 0.01)) - torch.log(0.01)) / (torch.log(1.01) - torch.log(0.01))

def log_np_norm(image):
    return (np.log(torch.add(image, 0.01)) - np.log(0.01)) / (np.log(1.01) - np.log(0.01))


def reconstruct_output(inputs):
        normal = inputs[:, 0:3, :, :]
        diffuse = inputs[:, 3:6, :, :]
        roughness = torch.cat([inputs[:, 6:7, :, :], inputs[:, 6:7, :, :], inputs[:, 6:7, :, :]], dim=1)
        specular = inputs[:, 7:, :, :]

        return normal, diffuse, roughness, specular

