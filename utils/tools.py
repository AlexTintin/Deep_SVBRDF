from __future__ import division
import math
import random
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from time import time

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from sklearn import manifold
from sklearn.utils import check_random_state

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

def save_image(image_init,inputs,rendered,legend,plot):
    normal = inputs[:, 0:3, :, :]
    diffuse = inputs[:, 3:6, :, :]
    roughness = torch.cat([inputs[:, 6:7, :, :], inputs[:, 6:7, :, :], inputs[:, 6:7, :, :]], dim=1)
    specular = inputs[:, 7:, :, :]
    img = torch.cat([image_init,normal,diffuse,roughness,specular,rendered],axis = 3)
   # img = img / 2 + 0.5     # unnormalize
    npimg = img.squeeze(0).cpu().detach().numpy()
    print(np.shape(npimg))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if plot:
        plt.show()
    plt.savefig(legend+'.png')

def tsne(latent,i):
    # Author: Jaques Grobler <jaques.grobler@inria.fr>
    # License: BSD 3 clause

    # Perform t-distributed stochastic neighbor embedding.
    t0 = time()
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    trans_data = tsne.fit_transform(latent).T
    t1 = time()
    print("t-SNE: %.2g sec" % (t1 - t0))
    plt.scatter(trans_data[0], trans_data[1])
    plt.title("t-SNE (%.2g sec)" % (t1 - t0))

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
   # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

