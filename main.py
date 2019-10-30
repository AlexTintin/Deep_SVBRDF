from utils import config
import glob, os
import torch
import matplotlib.pyplot as plt
from src import dataloader
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from src.model_AE_Linear import *
from src.model_AE import *
from src.Model_Unet import *
from src.Modified_Unet import *
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
from src.trainer import train_model

# Device = cpu ou cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#load les parametres dasn un dictionnaire
config = config()
# écrire les logs dans writer pour tensorboard
writer = SummaryWriter(config.path.logs_tensorboard)
print()
print("Use Hardware : ", device)
#Reproductibilites
random.seed(config.general.seed)
np.random.seed(config.general.seed)
torch.manual_seed(config.general.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Load data
print()
print("Load data")
# Transforme qui transforme en tensor
trans_all = transforms.Compose([
        transforms.ToTensor()
    ])
# Charger les dataloaders des 3 phases train / val / test
dataload_train = dataloader.Dataloader(config, phase = "train", transform=trans_all)
dataloadered_train = DataLoader(dataload_train, batch_size=config.train.batch_size,
                        shuffle=True, num_workers=config.train.num_workers)
dataload_val = dataloader.Dataloader(config, phase = "val", transform=trans_all)
dataloadered_val = DataLoader(dataload_val, batch_size=config.train.batch_size,
                        shuffle=True, num_workers=config.train.num_workers)
dataload_test = dataloader.Dataloader(config, phase = "test", transform=trans_all)
dataloadered_test = DataLoader(dataload_test, batch_size=config.train.batch_size,
                        shuffle=True, num_workers=config.train.num_workers)
# Dataloaders est 1 dictionnaire qui contient les 3 dataloaders
dataloaders = {'train': dataloadered_train, 'val': dataloadered_val, 'test':dataloadered_test}
print("End Load data")
print()
# Charger le model
print("Load model")
net = Unet() #Unetglobal() #autoencoder()
net.to(device)
# criterion : mean square error
criterion = nn.L1Loss()
#optimizer of Adam
optimizer = torch.optim.Adam(net.parameters(), lr=config.train.learning_rate,
weight_decay=config.train.weight_decay)
print("End Load model")
print()
print("Start training model")
print()
os.chdir('../../../')
# lancer l'entrainement du model
best_model = train_model(config, writer, net, dataloaders, criterion, optimizer, device)
print('Finished Training')
