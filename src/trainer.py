import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import copy
import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from src import dataloader
from torchvision import transforms, utils
from src.rendering_loss import *
import torchvision
import matplotlib.pyplot as plt
import time

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


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
    #model.load_state_dict(torch.load(config.path.load_path, map_location=torch.device('cpu')))
    best_model_wts = copy.deepcopy(model.state_dict())
    #introduction du best_loss pour le val, pour retenir le meilleur model
    best_loss = 100000
    n_batches = config.train.batch_size
    num_epochs = config.train.num_epochs
    learning_rate = config.train.learning_rate
    m=500
    #eps = (torch.empty((2, 512, 8, 8)).normal_(mean=0, std=0.2)).to(device)

    if config.train.loss == 'rendering' or config.train.loss == 'deep_rendering':
        rendering = True
    else:
        rendering = False

    #début de l'entrainement
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        if epoch % m == 0:
            print('lr is changing')
            learning_rate /= 2
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                         weight_decay=0.0000000000001)
            if epoch==1500:
                m=4000

        # Each epoch has a training and validation phase
        for phase in ['train']:#, 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            #initialisation des variables
            running_loss = 0.0
            running_corrects = 0
            nbre_sample = 0
            loss = 0

            # rendering loss init light and viewing
            if rendering :
                list_light, list_view = get_wlvs_np(256, 9)
            # Iterate over data.
            for index_data, data in enumerate(dataloaders[phase]):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data["input"].float().to(device), data["label"].float().to(device)
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    #x_latent = model.encode(inputs)
                    #Dze = model.decode(x_latent+eps)
                    #Dz = model.decode(x_latent)
                    outputs = model(inputs)
                    #loss_smooth = 2*L1Loss(Dz,Dze)
                    if rendering:
                    # rendering loss iterate over 10 different light and view positions
                        for j in range(9):
                            viewlight = list_light[j]
                            A = render(outputs, viewlight[1], viewlight[0], roughness_factor=0.0)
                            B = render(labels, viewlight[1], viewlight[0],roughness_factor=0.0)
                           # matplotlib_imshow(torchvision.utils.make_grid(B.detach()), one_channel=False)
                           # plt.show()
                            if config.train.loss == 'rendering':
                                loss += L1LogLoss(A,B)
                            else:
                                loss += criterion.lossVGG16_l1(A.to(device), B.to(device))
                    elif config.train.loss == 'l1':
                        loss = criterion(outputs, labels)
                    else:
                        loss = criterion.lossVGG16_l1test(outputs, labels)
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
            if epoch_loss < best_loss: #phase == 'val'  and  (epoch_loss < best_loss):
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





def train_model_rendring_loss(config, writer, model, dataloaders, criterion, optimizer, device):
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
    model.load_state_dict(torch.load(config.path.load_path + 'l1_15000', map_location=torch.device('cpu')))
    #model.load_state_dict(torch.load(config.path.load_path, map_location=torch.device('cpu')))
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
        for phase in ['train']:#, 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            #initialisation des variables
            #running_loss1 = 0.0
            #running_loss2 = 0.0
            #running_loss3 = 0.0
            #running_loss4 = 0.0
            running_loss = 0.0
            running_corrects = 0
            nbre_sample = 0
            # Iterate over data.
            # rendering loss init light and viewing
            # list_light, list_view = get_wlvs_np(256, 10)
            for index_data, data in enumerate(dataloaders[phase]):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data["input"].float().to(device), data["label"].float().to(device)
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    #outputs, mean, var = model(inputs)
                    #outputs = model(inputs,torch.mean(torch.mean(inputs, dim=2),dim=2))
                    outputs = model(inputs)
                    #loss = 0.0
                    #rendering loss iterate over 10 different light and view positions
                    #for j in range(10):
                     #   viewlight = list_light[j]
                      #  A = render(outputs, viewlight[1], viewlight[0], roughness_factor=0.0)
                       # B = render(labels, viewlight[1], viewlight[0],roughness_factor=0.0)
                        #loss+=L1LogLoss(A,B)
                        #matplotlib_imshow(torchvision.utils.make_grid(B), one_channel=False)
                        #plt.show()
                    #loss += criterion.lossVGG16_l1(A.to(device), B.to(device))
                    #loss/=10
                    #take the mean loss
                    #loss = loss4/5
                    loss = criterion.lossVGG16_l1(outputs, labels)
                   # loss3 = criterion2.lossVGG16_l1(outputs, labels)
                    #loss4 = criterion2.lossVGG16_rendering(A, B)
                    #loss = loss3
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * n_batches
                #running_loss4 += loss4 * n_batches
                #running_loss4 += loss4.item() * n_batches
                #running_loss3 += loss3.item() * n_batches
                #running_loss4 += loss4.item() * n_batches
                nbre_sample += n_batches
            epoch_loss = running_loss / nbre_sample
            #epoch_loss4 = running_loss4 / nbre_sample
            #epoch_loss2 = running_loss2 / nbre_sample
            #epoch_loss3 = running_loss3 / nbre_sample
           # epoch_loss4 = running_loss4 / nbre_sample
            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))
           # writer.add_scalar(phase + ' loss_deep_rendering',
            #                epoch_loss4,
            #                epoch)
           # writer.add_scalar(phase + ' loss_rendering',
            #                  epoch_loss1,
             #                 epoch)
           # writer.add_scalar(phase + ' loss_deep_l1',
            #                  epoch_loss3,
             #                 epoch)
          #  writer.add_scalar(phase + ' loss_deep_rendering',
           #                   epoch_loss4,
            #                  epoch)
            writer.add_scalar(phase + ' loss',
                              epoch_loss,
                              epoch)
            # deep copy the model
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
        '''
        if epoch==14999:
            print()
            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            print('Best val Loss: {:4f}'.format(best_loss))
            model.load_state_dict(best_model_wts)
            # save
            torch.save(model.state_dict(), config.path.result_path_model + 'l1_15000')
            best_loss =10000
        '''
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
                    phase, best_loss))
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





