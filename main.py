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
from src.VAE import *
from src.DisentAEmixUnet import *
from src.DisentAE import *
from src.Modified_Unet import *
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
from src.trainer import *
from src.VGG16Loss import *

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

#os.chdir("../")

dataload_test = dataloader.Dataloader(config, phase = "test", transform=trans_all)
dataloadered_test = DataLoader(dataload_test, batch_size=config.train.batch_size,
                        shuffle=True, num_workers=config.train.num_workers)

dataload_val = dataloader.Dataloader(config, phase = "val", transform=trans_all)
dataloadered_val = DataLoader(dataload_val, batch_size=config.train.batch_size,
                        shuffle=True, num_workers=config.train.num_workers)

# Charger le model
print("Load model")
net = DUnet()
net.to(device)

# criterion : see config
if config.train.loss == 'l1' or config.train.loss == 'rendering':
    criterion =  nn.L1Loss()
else:
    criterion = VGG16loss(device)

#optimizer of Adam
optimizer = torch.optim.Adam(net.parameters(), lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay)

print("End Load model")
print()


if config.train.real_training==False:

    dataload_train = dataloader.Dataloader(config, phase = "train",period = 'main', transform=trans_all)
    dataloadered_train = DataLoader(dataload_train, batch_size=config.train.batch_size,
                                    shuffle=True, num_workers=config.train.num_workers)
    dataloaders = {'train': dataloadered_train, 'val': dataloadered_val, 'test':dataloadered_test}
    print("End Load data")
    print()

    print("Start training model")
    print()
    """
    if config.train.rendering_loss:
    # lancer l'entrainement du model
        best_model = train_model_rendring_loss(config, writer, net, dataloaders, criterion, optimizer, device)
    else:
    """
    best_model = train_model(config, writer, net, dataloaders, criterion, optimizer, device)
    print('Finished Training')


else:
    print("Start training model")
    print()
    os.chdir('../')
    # lancer l'entrainement du model
    model= train_model_full(config, writer, net, dataloadered_val, criterion,optimizer, device)

    print('Finished Training')
