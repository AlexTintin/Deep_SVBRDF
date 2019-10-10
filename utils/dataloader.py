from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import glob, os
import pandas as pd



class Dataloader(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, config, phase, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        data_dico = {}
        self.phase = phase
        if phase == "train":
            data_path = config.path.data_path_train
        if phase == "val":
            data_path = config.path.data_path_val
        if phase == "test":
            data_path = config.path.data_path_test
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
        return len(self.dico)-1

    def __getitem__(self, idx):
        img_name = self.dico[idx]["nom_file"]
        image = io.imread(img_name)
        input = image[:,:288,:]
        normals = image[:,288:2*288,:]
        diffuse = image[:,2*288:3 * 288,:]
        roughness = image[:,3*288:4 * 288,:]
        specular = image[:,4*288:5 * 288,:]
        #label = np.concatenate((normals,diffuse,roughness,specular),axis = 2)
        sample = {'input': input, 'label': normals}
        if self.transform:
            sample = self.transform(sample)
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['input'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        label = label.transpose((2, 0, 1))
        return {'input': torch.from_numpy(image),
                'label': torch.from_numpy(label)}
