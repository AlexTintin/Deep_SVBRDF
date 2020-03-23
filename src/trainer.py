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
from src.FineGAN import *
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


def train_GAN(config, writer, dataloaders, device):
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

    # Modal options
#    __C.GAN = edict()
#    __C.GAN.DF_DIM = 64
#    __C.GAN.GF_DIM = 64
#    __C.GAN.Z_DIM = 100
#    __C.GAN.NETWORK_TYPE = 'default'
#    __C.GAN.R_NUM = 2
    HARDNEG_MAX_ITER = 1500

    torch.cuda.empty_cache()
    #chronomètre
    since = time.time()
    netG, netsD, num_Ds, start_count = load_network()
    avg_param_G = copy_G_params(netG)

    optimizerG, optimizersD = \
        define_optimizers(netG, netsD)

    num_batches = len(dataloaders["train"])
    criterion = nn.BCELoss(reduce=False)
    criterion_one = nn.BCELoss()
    criterion_class = nn.CrossEntropyLoss()
    nz = 100
    noise = Variable(torch.FloatTensor(config.batch_size, nz))
    fixed_noise = \
        Variable(torch.FloatTensor(config.batch_size, nz).normal_(0, 1))
    hard_noise = \
        Variable(torch.FloatTensor(config.batch_size, nz).normal_(0, 1))
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()#

    patch_stride = float(4)  # Receptive field stride given the current discriminator architecture for background stage
    n_out = 24  # Output size of the discriminator at the background stage; N X N where N = 24
    recp_field = 34  # Receptive field of each of the member of N X N

    print("Starting normal FineGAN training..")
    count = start_count
    start_epoch = start_count // (num_batches)

    #copier le meilleur model
    #model.load_state_dict(torch.load(config.load_path, map_location=torch.device('cpu')))
    #best_model_wts = copy.deepcopy(model.state_dict())
    #introduction du best_loss pour le val, pour retenir le meilleur model
    best_loss = 100000
    n_batches = config.batch_size
    num_epochs = config.num_epochs
    learning_rate = config.learning_rate
    lr_kl = 0.0002


    #C_max = Variable(cuda(torch.FloatTensor([config.C_max]), True))
    #lambda1 = lambda epoch: (-config.learning_rate / (100000)) * epoch + config.learning_rate
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    #eps = (torch.empty((2, 512, 8, 8)).normal_(mean=0, std=0.2)).to(device)


    if config.loss == 'rendering' or config.loss == 'deep_rendering':
        rendering = True
    else:
        rendering = False
    FineGan = FineGAN_trainer(config.result_path_model, dataloaders["train"])
    #début de l'entrainement
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        phase="train"
        #initialisation des variables
        running_loss = 0.0
        beta_vae_loss = 0
        nbre_sample = 0
        loss = 0

        # rendering loss init light and viewing
        if rendering :
            list_light, list_view = get_wlvs_np(256, 9)
        # Iterate over data.
        #if phase == 'train':
         #   scheduler.step()

        for step, data in enumerate(dataloaders[phase]):
            inputsx, inputsy, labels = data["inputx"].float().to(device), data["inputy"].float().to(device), data[
                "label"].float().to(device)
            # get the inputs; data is a list of [inputs, labels]

            rand_class = 2  # Randomly generating child code during training
            c_code = torch.zeros([200 ])
            c_code[rand_class] = 1
            c_code = Variable(c_code).cuda()

            print(c_code.size())
            ratio = 200 / 20
            arg_parent = torch.argmax(c_code) / ratio
            parent_c_code = torch.zeros([1, 20]).cuda()
            parent_c_code[0][int(arg_parent)] = 1
            p_code = parent_c_code

        # Feedforward through Generator. Obtain stagewise fake images
            noise.data.normal_(0, 1)

            print(noise.size(),p_code.size())
            fake_imgs = netG(noise,c_code,p_code) #I need to give the latent space here mixed with the noise find a way !!!!

            errD_total = 0

            flag = count % 100
            #batch_size = config.batch_size
            #criterion, criterion_one = self.criterion, self.criterion_one

            netD, optD = netsD[0], optimizersD[0]
            real_imgs = inputsy

            #fake_imgs = self.fake_imgs[0]
            netD.zero_grad()
            real_logits = netD(real_imgs)
            print(real_imgs.size(),fake_imgs[0].size())

            fake_labels = torch.zeros_like(real_logits[1])
            real_labels = torch.ones_like(real_logits[1])

            fake_logits = netD(fake_imgs[0])

            print(fake_labels.size(),fake_logits[1].size())
            errD_real = criterion_one(real_logits[1], real_labels)  # Real/Fake loss for the real image
            errD_fake = criterion_one(fake_logits[1], fake_labels)  # Real/Fake loss for the fake image
            errD = errD_real + errD_fake

            errD.backward()
            optD.step()

            '''if (flag == 0):
                summary_D = torch.summary.scalar('D_loss%d', errD.item())
                writer.add_summary(summary_D, count)
                summary_D_real = torch.summary.scalar('D_loss_real_%d', errD_real.item())
                writer.add_summary(summary_D_real, count)
                summary_D_fake = torch.summary.scalar('D_loss_fake_%d', errD_fake.item())
                writer.add_summary(summary_D_fake, count)
            '''

            errD_total += errD

            # Update the Generator networks
            netG.zero_grad()
            for myit in range(len(netsD)):
                netsD[myit].zero_grad()

            errG_total = 0
            flag = count % 100

            for i in range(num_Ds):

                outputs = netsD[i](fake_imgs[i])

                # real/fake loss for background (0) and child (2) stage
                real_labels = torch.ones_like(outputs[1])
                errG = criterion_one(outputs[1], real_labels)
                errG_total = errG_total + errG

                pred_c = netsD[i](fake_imgs[i])
                errG_info = criterion_class(pred_c[0], torch.nonzero(c_code.long())[0])

                errG_total = errG_total + errG_info

            errG_total.backward(retain_graph=True)
            for myit in range(len(netsD)):
                optimizerG[myit].step()


            for p, avg_p in zip(netG.parameters(), avg_param_G):
                avg_p.mul_(0.999).add_(0.001, p.data)

            count = count + 1


        #print(optimizer.state_dict()["param_groups"][0]["lr"])
        time_elapsed = time.time() - since
        print('epoch time complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        #epoch_loss = running_loss / nbre_sample
        #print('{} Loss: {:.4f}'.format(
        #    phase, epoch_loss))
        #writer.add_scalar(phase + ' loss',
        #                epoch_loss,
        #                epoch)
        # deep copy the model
        #if epoch_loss < best_loss:#phase == 'val'  and  (epoch_loss < best_loss):
        #    best_loss = epoch_loss
        #   best_model_wts = copy.deepcopy(model.state_dict())
    print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
    # load best model weights
    #model.load_state_dict(best_model_wts)
    #save
    #torch.save(model.state_dict(), config.result_path_model)
    save_model(netG, avg_param_G, netsD, count, config.result_path_model)

    print("Done with the normal training. Now performing hard negative training..")
    count = 0
    start_t = time.time()
    for step, data in enumerate(dataloaders[phase]):

        inputsx, inputsy, labels = data["inputx"].float().to(device), data["inputy"].float().to(device), data[
            "label"].float().to(device)

        rand_class = random.sample(range(200),
                                   1);  # Randomly generating child code during training
        c_code = torch.zeros([200, ])
        c_code[rand_class] = 1

        if (count % 2) == 0:  # Train on normal batch of images

            # Feedforward through Generator. Obtain stagewise fake images
            noise.data.normal_(0, 1)
            fake_imgs = netG(noise, c_code)

            #self.p_code = child_to_parent(self.c_code, cfg.FINE_GRAINED_CATEGORIES, cfg.SUPER_CATEGORIES)

            # Update discriminator networks
            errD_total = 0
            errD = FineGan.train_Dnet( count)
            errD_total += errD

            # Update the generator network
            errG_total = FineGan.train_Gnet(count)

        else:  # Train on degenerate images
            repeat_times = 10
            all_hard_z = Variable(torch.zeros(config.batch_size * repeat_times, nz)).cuda()
            all_hard_class = Variable(torch.zeros(config.batch_size * repeat_times, 200)).cuda()
            all_logits = Variable(torch.zeros(config.batch_size * repeat_times, )).cuda()

            for hard_it in range(repeat_times):
                hard_noise = hard_noise.data.normal_(0, 1)
                hard_class = Variable(torch.zeros([config.batch_size, 200])).cuda()
                my_rand_id = []

                for c_it in range(config.batch_size):
                    rand_class = random.sample(range(200), 1);
                    hard_class[c_it][rand_class] = 1
                    my_rand_id.append(rand_class)

                all_hard_z[config.batch_size * hard_it: config.batch_size * (hard_it + 1)] = hard_noise.data
                all_hard_class[config.batch_size * hard_it: config.batch_size * (hard_it + 1)] = hard_class.data
                fake_imgs = netG(hard_noise.detach(),hard_class.detach())

                fake_logits = netsD[0](fake_imgs[1].detach())
                smax_class = softmax(fake_logits[0], dim=1)

                for b_it in range(config.batch_size):
                    all_logits[(config.batch_size * hard_it) + b_it] = smax_class[b_it][my_rand_id[b_it]]

            sorted_val, indices_hard = torch.sort(all_logits)
            noise = all_hard_z[indices_hard[0: config.batch_size]]
            c_code = all_hard_class[indices_hard[0: config.batch_size]]

            fake_imgs = netG(noise, c_code)

            #self.p_code = child_to_parent(self.c_code, cfg.FINE_GRAINED_CATEGORIES, cfg.SUPER_CATEGORIES)

            # Update Discriminator networks
            errD_total = 0
            errD = FineGan.train_Dnet(0, count)
            errD_total += errD

            # Update generator network
            errG_total = FineGan.train_Gnet(count)

        for p, avg_p in zip(netG.parameters(), avg_param_G):
            avg_p.mul_(0.999).add_(0.001, p.data)
        count = count + 1

        end_t = time.time()

        if (count % 100) == 0:
            print('''[%d/%d][%d]
                                             Loss_D: %.2f Loss_G: %.2f Time: %.2fs
                                          '''
                  % (count, HARDNEG_MAX_ITER, num_batches,
                     errD_total.data[0], errG_total.data[0],
                     end_t - start_t))

        if (count == HARDNEG_MAX_ITER):  # Hard negative training complete
            break

    save_model(netG, avg_param_G, netsD, count, config.result_path_model)
    writer.close()

    return model