import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import copy
import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from src import dataloader
from torchvision import transforms, utils

trans_all = transforms.Compose([
        transforms.ToTensor()
    ])

def train_model(config, writer, model, dataloaders, criterion, optimizer, device):
    """
    input :
        config: dictionnaire qui contient les fig
        writer: tensorboard
        model: net
        dataloaders: les dataloaders de phase train et val
        criterion: criterion
        optimizer: optimizer
        device: cpu/gpu

    output :
        model:
    """

    #chronomètre
    since = time.time()
    #copier le meilleur model
    best_model_wts = copy.deepcopy(model.state_dict())
    #introduction du best_loss pour le val, pour retenir le meilleur model
    best_loss = 100000
    n_batches = config.train.batch_size
    num_epochs = config.train.num_epochs
    #début de l'entrainement
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            #initialisation des variables
            running_loss = 0.0
            running_corrects = 0
            nbre_sample = 0
            # Iterate over data.
            for index_data, data in enumerate(dataloaders[phase]):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data["input"].float().to(device), data["label"].float().to(device)
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    #outputs, mean, var = model(inputs)
                    #outputs = model(inputs,torch.mean(torch.mean(inputs, dim=2),dim=2))
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)#-0.5 * torch.sum(1 + var - mean.pow(2) - var.exp())
                    #outputs = model(inputs, (torch.mean(torch.mean(inputs, dim=2),dim=2)))
                    #loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * n_batches
                nbre_sample += n_batches
            epoch_loss = running_loss / nbre_sample
            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))
            writer.add_scalar(phase + ' loss',
                            epoch_loss,
                            epoch)
            # deep copy the model
            if phase == 'val'  and  (epoch_loss < best_loss):
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))
    # load best model weights
    model.load_state_dict(best_model_wts)
    #save
    torch.save(model.state_dict(), config.path.result_path_model)
    return model


def train_model_init(config, writer, model, dataloaders, criterion, optimizer, device):
    """
    input :
        config: dictionnaire qui contient les fig
        writer: tensorboard
        model: net
        dataloaders: les dataloaders de phase train et val
        criterion: criterion
        optimizer: optimizer
        device: cpu/gpu

    output :
        model:
    """

    #chronomètre
    since = time.time()
    #copier le meilleur model
    #best_model_wts = copy.deepcopy(model.state_dict())
    #introduction du best_loss pour le val, pour retenir le meilleur model
    best_loss = 100000
    n_batches = config.train.batch_size
    num_epochs = config.train.num_epochs
    #début de l'entrainement
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            #initialisation des variables
            running_loss = 0.0
            running_corrects = 0
            nbre_sample = 0
            # Iterate over data.
            for index_data, data in enumerate(dataloaders[phase]):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data["input"].float().to(device), data["label"].float().to(device)
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    #outputs = model(inputs, (torch.mean(torch.mean(inputs, dim=2),dim=2)))
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase1
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * n_batches
                nbre_sample += n_batches
            epoch_loss = running_loss / nbre_sample
            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))
            writer.add_scalar(phase + ' loss',
                            epoch_loss,
                            epoch)
            # deep copy the model
            if phase == 'val'  and  (epoch_loss < best_loss):
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
        print()
    last_running_loss = running_loss
    last_model_weight = copy.deepcopy(model.state_dict())
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))
    # load best model weights
    model.load_state_dict(best_model_wts)
    #save
    torch.save(model.state_dict(), config.path.load_path)
    model.load_state_dict(last_model_weight)
    return model,best_loss, last_running_loss, best_model_wts,nbre_sample



def train_model_iter(config, writer, model, dataloaders, criterion, optimizer, device,best_loss, last_running_loss, best_model_w, nbre_sample):
    """
    input :
        config: dictionnaire qui contient les fig
        writer: tensorboard
        model: net
        dataloaders: les dataloaders de phase train et val
        criterion: criterion
        optimizer: optimizer
        device: cpu/gpu

    output :
        model:
    """

    # chronomètre
    since = time.time()
    # copier le meilleur model
    best_model_wts = best_model_w
    # introduction du best_loss pour le val, pour retenir le meilleur model
   # best_loss = 100000
    n_batches = config.train.batch_size
    num_epochs = config.train.num_epochs
    # début de l'entrainement
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            # initialisation des variables
            running_loss = last_running_loss
            #running_corrects = 0
            #nbre_sample = 0
            # Iterate over data.
            for index_data, data in enumerate(dataloaders[phase]):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data["input"].float().to(device), data["label"].float().to(device)
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # outputs = model(inputs, (torch.mean(torch.mean(inputs, dim=2),dim=2)))
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * n_batches
                nbre_sample += n_batches
            epoch_loss = running_loss / nbre_sample
            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))
            writer.add_scalar(phase + ' loss',
                              epoch_loss,
                              epoch)
            # deep copy the model
            if  phase == 'val'  and (epoch_loss < best_loss):
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
        print()
    last_running_loss = running_loss
    last_model_weight = copy.deepcopy(model.state_dict())
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))
    # load best model weights
    model.load_state_dict(best_model_wts)
    # save
    torch.save(model.state_dict(), config.path.load_path)
    model.load_state_dict(last_model_weight)
    return model,best_loss, last_running_loss, best_model_wts,nbre_sample


def train_model_full(config, writer, model, dataloadered_val, criterion, optimizer, device):
    """
    input :
        config: dictionnaire qui contient les fig
        writer: tensorboard
        model: net
        dataloaders: les dataloaders de phase train et val
        criterion: criterion
        optimizer: optimizer
        device: cpu/gpu

    output :
        model:
    """

    # chronomètre
    since = time.time()
    # copier le meilleur model
    best_model_wts = copy.deepcopy(model.state_dict())
    #introduction du best_loss pour le val, pour retenir le meilleur model
    best_loss = 100000
    n_batches = config.train.batch_size
    num_epochs = config.train.num_epochs
    # début de l'entrainement
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        running_loss = 0.0
        #running_corrects = 0
        nbre_sample = 0
        phase = 'train'
        # Iterate over data.
        for i in range(config.train.trainset_division):
            dataload_train = dataloader.Dataloader(config, phase="train", iteration=i, period='main',
                                                       transform=trans_all)
            dataloadered_train = DataLoader(dataload_train, batch_size=config.train.batch_size,
                                               shuffle=True, num_workers=config.train.num_workers)
            dataloaders = {'train': dataloadered_train, 'val': dataloadered_val}
            if ((i%100==0) & (i!=0)):
                phase = 'val'
                model.eval()
            for index_data, data in enumerate(dataloaders[phase]):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data["input"].float().to(device), data["label"].float().to(device)
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # outputs = model(inputs, (torch.mean(torch.mean(inputs, dim=2),dim=2)))
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * n_batches
                nbre_sample += n_batches
            current_loss = running_loss / nbre_sample
            print('{} Loss: {:.4f}'.format(
                i, current_loss))
            if phase == 'val' and (current_loss < best_loss):
                best_loss = current_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                model.load_state_dict(best_model_wts)
                # save
                torch.save(model.state_dict(), config.path.load_path)
            if phase=='val':
                print('{} Loss: {:.4f}'.format(
                    phase, current_loss))
                i-=1
                writer.add_scalar(phase + ' loss',
                                  current_loss,
                                  epoch)
                phase = 'train'

        epoch_loss = running_loss / nbre_sample
        print('{} Loss: {:.4f}'.format(
            phase, epoch_loss))
        writer.add_scalar(phase + ' loss',
                          epoch_loss,
                          epoch)
        # deep copy the model
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))
    # load best model weights
    model.load_state_dict(best_model_wts)
    # save
    torch.save(model.state_dict(), config.path.load_path)
    return model





