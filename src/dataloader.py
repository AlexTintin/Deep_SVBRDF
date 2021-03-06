from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from skimage import io
from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms, utils
import glob, os

from src.rendering_loss import *
import torchvision
import matplotlib.pyplot as plt
from skimage.transform import resize


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

class Dataloader(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, config, phase,iteration=0, period='nul', transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        data_dico = {}
        self.phase = phase
        self.iter = iteration
        self.period = period
        if(phase == "train"):
            data_path = config.data_path_train
            if self.iter == 0:
                os.chdir(data_path)
        if phase == "val":
            data_path = config.data_path_val
            os.chdir(data_path)
        if phase == "test":
            data_path = config.data_path_test
            os.chdir(data_path)
        for index_file, file in enumerate(glob.glob("*.png")):
            split_name = file.split(";")
            data_dico[index_file] = {'nom_file': data_path+"/"+file, "id": int(split_name[0]),
                                             "texture": split_name[1].split(("_")),
                                             "info_inconnue": split_name[1].split(("."))[0]}
        self.dico =  data_dico
        self.config = config
        self.transform = transform

    def __len__(self):
            return (len(self.dico))

    def __getitem__(self, idx):
        img_name = self.dico[idx]["nom_file"]
        image = io.imread(img_name)
        delta = int(32/2)
        input = (np.log(image[delta:-delta,delta:288-delta,:]/255+0.01)-np.log(0.01))/(np.log(1.01)-np.log(0.01))
        normals = (image[delta:-delta,288+delta:2*288-delta,:]/255)
        diffuse = (image[delta:-delta,2*288+delta:3 * 288-delta,:]/255)
        roughness = (image[delta:-delta,3*288+delta:4 * 288-delta,0]/255)
        specular = (image[delta:-delta,4*288+delta:5 * 288-delta,:]/255)
        label = np.concatenate((diffuse,np.expand_dims(roughness, axis=2),specular),axis = 2)
        #label = np.concatenate((diffuse, roughness, specular), axis=2)
        #label *=2
        #label -=1
        if self.transform:
            input_t = self.transform(input)
            label_t = self.transform(label)
            normals_t = self.transform(normals)
            roughness_t = self.transform(roughness)
            sample = {'inputx': input_t, 'inputy': normals_t,'label': label_t}
        else:
            sample = {'inputx': input, 'inputy': normals,'label': label}
        return sample