from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import glob, os



class Dataloader(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, config, phase,iteration=0, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        data_dico = {}
        self.phase = phase
        self.totiter = config.train.trainset_division
        self.iter = iteration
        self.realdata = config.train.real_training
        if ((phase == "train") & (iteration==0)):
            data_path = config.path.data_path_train
            os.chdir('../../../'+data_path)
        elif(phase == "train"):
            data_path = config.path.data_path_train
            os.chdir(data_path)
        if phase == "val":
            data_path = config.path.data_path_val
            os.chdir(data_path)
        if phase == "test":
            data_path = config.path.data_path_test
            os.chdir('../../../'+data_path)
        for index_file, file in enumerate(glob.glob("*.png")):
            split_name = file.split(";")
            data_dico[index_file] = {'nom_file': data_path+"/"+file, "id": int(split_name[0]),
                                             "texture": split_name[1].split(("_")),
                                             "info_inconnue": split_name[1].split(("."))[0]}
        self.dico =  data_dico
        self.config = config
        self.transform = transform

    def __len__(self):
        if ((self.phase == "train") & self.realdata==True):
            return int((len(self.dico)-1)/self.totiter)
        else:
            return (len(self.dico) - 1)

    def __getitem__(self, idx):
        if ((self.phase == "train") & self.realdata==True):
            img_name = self.dico[int((len(self.dico)-1)/self.totiter)*self.iter+idx]["nom_file"]
        else:
            img_name = self.dico[idx]["nom_file"]
        image = io.imread(img_name)
        input = image[:,:288,:]
        normals = image[:,288:2*288,:]
        diffuse = image[:,2*288:3 * 288,:]
        roughness = image[:,3*288:4 * 288,:]
        specular = image[:,4*288:5 * 288,:]
        #label = np.concatenate((normals,diffuse,roughness,specular),axis = 2)
        if self.transform:
            input_t = self.transform(input)
            normals_t = self.transform(normals)
            sample = {'input': input_t, 'label': normals_t}
        else:
            sample = {'input': input, 'label': normals}
        return sample
