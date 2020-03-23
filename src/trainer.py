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
from torch.autograd import Variable
from src.rendering_loss import *
import torchvision
import matplotlib.pyplot as plt
import time

def cuda(tensor, uses_cuda):
    return tensor.cuda() if uses_cuda else tensor

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
    #free some memory
    torch.cuda.empty_cache()

    #chronomètre
    since = time.time()

    #copier le meilleur model
    #model.load_state_dict(torch.load(config.load_path, map_location=torch.device('cpu')))
    #best_model_wts = copy.deepcopy(model.state_dict())

    #introduction du best_loss pour le val, pour retenir le meilleur model
    best_loss = 100000
    n_batches = config.batch_size
    num_epochs = config.num_epochs
    #kl_loss_coef = 0.0004
    #scheduler for adaptive leaarning rate
    #lambda1 = lambda epoch: (-config.learning_rate / (100000)) * epoch + config.learning_rate
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    #loss function
    if config.loss == 'rendering' or config.loss == 'deep_rendering':
        rendering = True
    else:
        rendering = False


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
            nbre_sample = 0
            loss = 0

            # rendering loss init light and viewing
            if rendering :
                list_light, list_view = get_wlvs_np(256, 9)

            # Iterate over data.
            for index_data, data in enumerate(dataloaders[phase]):
                # get the inputs; data is a list of [inputs, labels]
                inputsx,inputsy, labels = data["inputx"].float().to(device),data["inputy"].float().to(device), data["label"].float().to(device)
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs,z,x_latent,z2,x_latent2 = model(inputsx,inputsy)
                    #calculate kl loss from unet dissentangled paper
                    #klloss = model.latent_kl(z, x_latent)
                    #klloss2 = model.latent_kl(z2, x_latent2)

                    if rendering:
                        normal = inputsy
                    # rendering loss iterate over 10 different light and view positions
                        for j in range(9):
                            viewlight = list_light[j]
                            A = render(torch.cat([normal,outputs],dim=1), viewlight[1], viewlight[0], roughness_factor=0.0)
                            B = render(torch.cat([normal,labels],dim=1), viewlight[1], viewlight[0],roughness_factor=0.0)
                           # matplotlib_imshow(torchvision.utils.make_grid(B.detach()), one_channel=False)
                           # plt.show()
                            if config.loss == 'rendering':
                                loss += L1LogLoss(A,B)
                            else:
                                loss += criterion.lossVGG16_l1(A.to(device), B.to(device))

                    elif config.loss == 'l1':
                        loss = criterion(outputs, labels)
                    else:
                        loss = criterion.VGG19Loss(outputs, labels)

                    vae_loss = loss #+ kl_loss_coef*(klloss+klloss2)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        vae_loss.backward()
                        optimizer.step()
                        #scheduler.step()
                # statistics
                running_loss += vae_loss.item() * n_batches
                nbre_sample += n_batches
            if phase == 'train':
                print(optimizer.state_dict()["param_groups"][0]["lr"])
                time_elapsed = time.time() - since
                print('epoch time complete in {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))
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
    torch.save(model.state_dict(), config.result_path_model)
    return model
