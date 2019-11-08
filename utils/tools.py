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
    Length = torch.sqrt(torch.sum(t*t, axis=-1, keepdims=True))
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


def reconstruct_output(outputs, order='NDRS'):
    with torch.variable_scope("reconstruct_output"):
        if order == 'NDRS':
            partial_normal = outputs[:, :, :, 0:2]
            diffuse = outputs[:, :, :, 2:5]
            roughness = outputs[:, :, :, 5:6]
            specular = outputs[:, :, :, 6:9]
        elif order == 'DSRN':
            partial_normal = outputs[:, :, :, 7:9]
            diffuse = outputs[:, :, :, 0:3]
            roughness = outputs[:, :, :, 6:7]
            specular = outputs[:, :, :, 3:6]

        normal_shape = torch.shape(partial_normal)
        normal_z = torch.ones([normal_shape[0], normal_shape[1], normal_shape[2], 1], torch.float32)
        normal = tensor_norm(torch.concat([partial_normal, normal_z], axis=-1))

        outputs_final = torch.concat([normal, diffuse, roughness, roughness, roughness, specular], axis=-1)
        return outputs_final
