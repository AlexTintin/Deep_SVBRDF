from utils import config, ToTensor
import glob, os
import torch
import matplotlib.pyplot as plt
from utils import dataloader
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from src.model_AE import *
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
from src.trainer import train_model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = config()
writer = SummaryWriter(config.path.logs_tensorboard)
print()
print("Use Hardware : ", device)
#Reproductibilites
random.seed(config.general.seed)
np.random.seed(config.general.seed)
torch.manual_seed(config.general.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#Load data
print()
print("Load data")
dataload_train = dataloader.Dataloader(config, phase = "train", transform=transforms.Compose([ToTensor()]))
dataloadered_train = DataLoader(dataload_train, batch_size=config.train.batch_size,
                        shuffle=True, num_workers=config.train.num_workers)
dataload_val = dataloader.Dataloader(config, phase = "val", transform=transforms.Compose([ToTensor()]))
dataloadered_val = DataLoader(dataload_val, batch_size=config.train.batch_size,
                        shuffle=True, num_workers=config.train.num_workers)
dataload_test = dataloader.Dataloader(config, phase = "test", transform=transforms.Compose([ToTensor()]))
dataloadered_test = DataLoader(dataload_test, batch_size=config.train.batch_size,
                        shuffle=True, num_workers=config.train.num_workers)
dataloaders = {'train': dataloadered_train, 'val': dataloadered_val, 'test':dataloadered_test}
#Load model
print("End Load data")
print()
print("Load model")
net = autoencoder()
net.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=config.train.learning_rate,
weight_decay=config.train.weight_decay)
print("End Load model")
print()
print("Start training model")
print()
best_model = train_model(config, writer, net, dataloaders, criterion, optimizer,device, num_epochs=config.train.num_epochs)
print('Finished Training')
"""



#sample = dataload[1]
#datasample = sample['input']
print(datasample.shape)
x_latent = net.encoder(datasample)
print(x_latent.shape)

def plot_slidder(x_latent, net):
    pass
x_latent_modify = x_latent + 1
sortie_to_plot = net.decoder(x_latent_modify)


fig = plt.figure()

for i in range(len(dataload)):
    sample = dataload[i+1]
    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    plt.imshow(sample["input"])

    if i == 3:
        plt.show()
        break


"""
