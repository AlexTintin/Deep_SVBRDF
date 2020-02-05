#from utils import config
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
from src.UnetDisentagled import *
from src.DisentAE import *
from src.Pretrain_encode import *
from src.AE_Goa import *
from src.Modified_Unet import *
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
import argparse
from src.trainer import *
from src.VGG16Loss import *


parser = argparse.ArgumentParser()
parser.add_argument("--data_path_realtrain",type=str, default="../DeepMaterialsData/train")
parser.add_argument("--data_path_train",type=str, default="../test")
parser.add_argument("--data_path_val",type=str, default="../val")
parser.add_argument("--data_path_test",type=str, default="./../new_dataset/test")
parser.add_argument("--result_path_model",type=str, default="./../../Deep_SVBRDF_local/content/DUNET_Resnet4.pt" )
parser.add_argument("--logs_tensorboard",type=str, default="../../runs/DUNET_Resnet4")
parser.add_argument("--load_path", type=str, default="../../Deep_SVBRDF_local/content/DUNET_Resnet4.pt")
parser.add_argument("--seed", type=int, default=0 )
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--learning_rate", type=float, default=0.00005)
parser.add_argument("--weight_decay", type=float, default=0.000000001 )
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--trainset_division", type=int, default=10000, help="scale images to this size before cropping to 256x256")
parser.add_argument("--real_training", type=bool, default=False)
parser.add_argument("--loss", type=str, default="rendering")

config = parser.parse_args()



# Device = cpu ou cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#load les parametres dasn un dictionnaire
#config = config()

# écrire les logs dans writer pour tensorboard
writer = SummaryWriter(config.logs_tensorboard)
print()
print("Use Hardware : ", device)

#Reproductibilites
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
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
dataloadered_test = DataLoader(dataload_test, batch_size=config.batch_size,
                        shuffle=True, num_workers=config.num_workers)

dataload_val = dataloader.Dataloader(config, phase = "val", transform=trans_all)
dataloadered_val = DataLoader(dataload_val, batch_size=config.batch_size,
                        shuffle=True, num_workers=config.num_workers)

# Charger le model
print("Load model")
net = DUnet()
net.to(device)

# criterion : see config
if config.loss == 'l1' or config.loss == 'rendering':
    criterion = nn.L1Loss()
else:
    criterion = Resfeat(device)

#optimizer of Adam
optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate,
                             weight_decay=config.weight_decay)

print("End Load model")
print()


if config.real_training==False:

    dataload_train = dataloader.Dataloader(config, phase = "train",period = 'main', transform=trans_all)
    dataloadered_train = DataLoader(dataload_train, batch_size=config.batch_size,
                                    shuffle=True, num_workers=config.num_workers)
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
