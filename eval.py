from utils import config, ToTensor
import glob, os
import torch
import matplotlib.pyplot as plt
from utils import dataloader
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from src.model_AE_Linear import *
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
from src.trainer import train_model
import torchvision

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

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
dataloadered_train = DataLoader(dataload_train, batch_size=4,
                        shuffle=True, num_workers=config.train.num_workers)
dataload_val = dataloader.Dataloader(config, phase = "val", transform=transforms.Compose([ToTensor()]))
dataloadered_val = DataLoader(dataload_val, batch_size=config.train.batch_size,
                        shuffle=True, num_workers=config.train.num_workers)
dataload_test = dataloader.Dataloader(config, phase = "test", transform=transforms.Compose([ToTensor()]))
dataloadered_test = DataLoader(dataload_test, batch_size=config.train.batch_size,
                        shuffle=True, num_workers=config.train.num_workers)
dataloaders = {'train': dataloadered_train, 'val': dataloadered_val, 'test':dataloadered_test}
#Load model

# get some random training images
dataiter = iter(dataloadered_train)
sample = dataiter.next()
images, labels = sample["input"], sample["label"]
# create grid of images
img_grid = torchvision.utils.make_grid(images)
img_grid_labels = torchvision.utils.make_grid(labels)
# show images
matplotlib_imshow(img_grid, one_channel=False)
matplotlib_imshow(img_grid_labels, one_channel=False)
# write to tensorboard
writer.add_image('4_images_of_dataset_input', img_grid)
writer.add_image('4_images_of_dataset_output', img_grid_labels)


print("End Load data")
print()
print("Load model")
the_model = AE_Linear()
the_model.to(device)
writer.add_graph(the_model, images.float().to(device))
writer.close()
the_model.load_state_dict(torch.load(config.path.result_path_model, map_location=torch.device('cpu')))
the_model.eval()


dataiter = iter(dataloadered_train)
sample = dataiter.next()
images, labels = sample["input"], sample["label"]
# create grid of images
img_grid = torchvision.utils.make_grid(images)
img_grid_labels = torchvision.utils.make_grid(labels)
# show images
matplotlib_imshow(img_grid, one_channel=False)
plt.show()
matplotlib_imshow(img_grid_labels, one_channel=False)
plt.show()
x_latent = the_model.encoder(images.float().to(device))
sortie_to_plot = the_model.decoder(x_latent.float().to(device))
img_grid_sortie_to_plot = torchvision.utils.make_grid(sortie_to_plot)
print(img_grid_sortie_to_plot.cpu().detach().numpy())
matplotlib_imshow(img_grid_sortie_to_plot.cpu().detach(), one_channel=False)
plt.show()
