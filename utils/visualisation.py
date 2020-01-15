
import torchvision
import io
import torch
import numpy as np

import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, models, transforms

import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn import datasets
import pandas as pd
from time import time
import numpy as np
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection, neighbors)
import umap
from PIL import Image
from os import walk


from utils import config
import glob, os
import torch
import matplotlib.pyplot as plt
from src import dataloader
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from src.Model_Unet import *
from src.VAE import *
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
from src.trainer import train_model
import torchvision
from utils.tools import *
from src.rendering_loss import *
from sklearn.decomposition import PCA
from utils.tools import *
import matplotlib.image as mpimg


# Device = cpu ou cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = config()

trans_all = transforms.Compose([
        transforms.ToTensor()
    ])


def plot_embedding(X, reds, blues, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure(figsize=(10, 5))
    plt.scatter(X[reds, 0], X[reds, 1], c="red",
                s=20, edgecolor='k')
    plt.scatter(X[blues, 0], X[blues, 1], c="blue",
                s=20, edgecolor='k')
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


def plot_embedding_5element(X5, reds5, blues5, title=None):
    for i in range(5):
        X = X5[i]
        reds = reds5[i]
        blues = blues5[i]
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)
        plt.figure(figsize=(10, 5))
        plt.subplot(5, 1, i + 1)
        plt.scatter(X[reds, 0], X[reds, 1], c="red",
                    s=20, edgecolor='k')
        plt.scatter(X[blues, 0], X[blues, 1], c="blue",
                    s=20, edgecolor='k')
        plt.xticks([]), plt.yticks([])
        if title is not None:
            plt.title(title)


def plot_pca(X_pca, pca, reds, blues, title=""):
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.scatter(X_pca[reds, 0], X_pca[reds, 1], c="red",
                s=20, edgecolor='k')
    plt.scatter(X_pca[blues, 0], X_pca[blues, 1], c="blue",
                s=20, edgecolor='k')
    plt.title("Projection by PCA " + title)
    plt.xlabel("1st principal component")
    plt.ylabel("2nd component")
    plt.subplot(1, 2, 2)
    ebouli = pd.Series(pca.explained_variance_ratio_)
    coef = np.transpose(pca.components_)
    cols = ['PC-' + str(x + 1) for x in range(len(ebouli))]
    pc_infos = pd.DataFrame(coef, columns=cols)
    plt.Circle((0, 0), radius=10, color='g', fill=False)
    circle1 = plt.Circle((0, 0), radius=1, color='g', fill=False)
    plt.gcf().gca().add_artist(circle1)
    for idx in range(10):
        x1 = pc_infos["PC-1"][idx]
        y1 = pc_infos["PC-2"][idx]
        plt.plot([0.0, x1], [0.0, y1], 'k-')
        plt.plot(x1, y1, 'rx')
        plt.annotate(pc_infos.index[idx], xy=(x1, y1))
    plt.xlabel("PC-1 (%s%%)" % str(ebouli[0])[:4].lstrip("0."))
    plt.ylabel("PC-2 (%s%%)" % str(ebouli[1])[:4].lstrip("0."))
    plt.xlim((-1, 1))
    plt.ylim((-1, 1))
    plt.title("Circle of Correlations")


def get_images_asdata(mypath, type_image_plot, label1, label2):
    X = []
    Y = []
    flag_X = False
    for (dirpath, dirnames, filenames) in walk(mypath):
        if "ipynb_checkpoints" not in dirpath:
            for filename in filenames:
                flag_X = False
                category = filename.split(";")[1].split("_")[0]
                if label1 in category:
                    Y.append(0)
                    flag_X = True
                if label2 in category:
                    Y.append(1)
                    flag_X = True
                if flag_X:
                    img = Image.open(dirpath + filename)
                    img.load()
                    data = np.asarray(img, dtype="float64")
                    X.append(data[:, 288 * type_image_plot:288 * (type_image_plot + 1), :])
    X = np.array(X)
    Y = np.array(Y)
    return (X, Y)


def plot_pca_5element(X_pcag5, pcag5, reds5, blues5, title=""):
    plt.figure(figsize=(20, 10))
    for position in range(5):
        X_pca = X_pcag5[position]
        pca = pcag5[position]
        reds = reds5[position]
        blues = blues5[position]
        plt.subplot(2, 5, position + 1)
        plt.scatter(X_pca[reds, 0], X_pca[reds, 1], c="red",
                    s=20, edgecolor='k')
        plt.scatter(X_pca[blues, 0], X_pca[blues, 1], c="blue",
                    s=20, edgecolor='k')
        plt.title("Projection by PCA " + title)
        plt.xlabel("1st principal component")
        plt.ylabel("2nd component")
        plt.subplot(2, 5, 5 + position + 1)
        ebouli = pd.Series(pca.explained_variance_ratio_)
        coef = np.transpose(pca.components_)
        cols = ['PC-' + str(x + 1) for x in range(len(ebouli))]
        pc_infos = pd.DataFrame(coef, columns=cols)
        plt.Circle((0, 0), radius=10, color='g', fill=False)
        circle1 = plt.Circle((0, 0), radius=1, color='g', fill=False)
        plt.gcf().gca().add_artist(circle1)
        for idx in range(10):
            x1 = pc_infos["PC-1"][idx]
            y1 = pc_infos["PC-2"][idx]
            plt.plot([0.0, x1], [0.0, y1], 'k-')
            plt.plot(x1, y1, 'rx')
            plt.annotate(pc_infos.index[idx], xy=(x1, y1))
        plt.xlabel("PC-1 (%s%%)" % str(ebouli[0])[:4].lstrip("0."))
        plt.ylabel("PC-2 (%s%%)" % str(ebouli[1])[:4].lstrip("0."))
        plt.xlim((-1, 1))
        plt.ylim((-1, 1))
    plt.title("Circle of Correlations")



n_neighbors = 30

#Reproductibilites
random.seed(config.general.seed)
np.random.seed(config.general.seed)
torch.manual_seed(config.general.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

mypath = "../../my_little_dataset/train/"

type_image_plot1 = 0 # L'image à considere : l'image original (0) / les normales (1) etc...
label11 = "brick" # 1er nom dans le label
label12 = "leather" # 1er nom dans le label

type_image_plot2 = 1 # L'image à considere : l'image original (0) / les normales (1) etc...
label21 = "brick" # 1er nom dans le label
label22 = "leather" # 1er nom dans le label

type_image_plot3 = 2 # L'image à considere : l'image original (0) / les normales (1) etc...
label31 = "brick" # 1er nom dans le label
label32 = "leather" # 1er nom dans le label

type_image_plot4 = 3 # L'image à considere : l'image original (0) / les normales (1) etc...
label41 = "brick" # 1er nom dans le label
label42 = "leather" # 1er nom dans le label

type_image_plot5 = 4 # L'image à considere : l'image original (0) / les normales (1) etc...
label51 = "brick" # 1er nom dans le label
label52 = "leather" # 1er nom dans le label



X1i, Y1 = get_images_asdata(mypath, type_image_plot1, label11, label12)
X1 = X1i.reshape(len(Y1),-1)
pca1 = decomposition.PCA(n_components=2)
pca1.fit(X1)
X_pca1 = pca1.transform(X1)
reds1 = Y1 == 0
blues1 = Y1 == 1
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
X_tsne1 = tsne.fit_transform(X1)
reducer = umap.UMAP(random_state=0)
embedding1 = reducer.fit_transform(X1)


X2, Y2 = get_images_asdata(mypath, type_image_plot2, label21, label22)
X2 = X2.reshape(len(Y2),-1)
pca2 = decomposition.PCA(n_components=2)
pca2.fit(X2)
X_pca2 = pca2.transform(X2)
reds2 = Y2 == 0
blues2 = Y2 == 1
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
X_tsne2 = tsne.fit_transform(X2)
reducer = umap.UMAP(random_state=0)
embedding2 = reducer.fit_transform(X2)

X3, Y3 = get_images_asdata(mypath, type_image_plot3, label31, label32)
X3 = X3.reshape(len(Y3),-1)
pca3 = decomposition.PCA(n_components=2)
pca3.fit(X3)
X_pca3 = pca3.transform(X3)
reds3 = Y3 == 0
blues3 = Y3 == 1
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
X_tsne3 = tsne.fit_transform(X3)
reducer = umap.UMAP(random_state=0)
embedding3 = reducer.fit_transform(X3)

X4, Y4 = get_images_asdata(mypath, type_image_plot4, label41, label42)
X4 = X4.reshape(len(Y4),-1)
pca4 = decomposition.PCA(n_components=2)
pca4.fit(X4)
X_pca4 = pca4.transform(X4)
reds4 = Y4 == 0
blues4 = Y4 == 1
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
X_tsne4 = tsne.fit_transform(X4)
reducer = umap.UMAP(random_state=0)
embedding4 = reducer.fit_transform(X4)

X5, Y5 = get_images_asdata(mypath, type_image_plot5, label51, label52)
X5 = X5.reshape(len(Y5),-1)
pca5 = decomposition.PCA(n_components=2)
pca5.fit(X5)
X_pca5 = pca5.transform(X5)
reds5 = Y5 == 0
blues5 = Y5 == 1
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
X_tsne5 = tsne.fit_transform(X5)
reducer = umap.UMAP(random_state=0)
embedding5 = reducer.fit_transform(X5)

X5element = np.array([X_pca1,X_pca2, X_pca3,X_pca4, X_pca5])
pca5element = np.array([pca1, pca2, pca3,pca4, pca5])
reds5element = np.array([reds1,reds2, reds3,reds4, reds5])
blues5element = np.array([blues1,blues2, blues3,blues4, blues5])

#fig = plot_pca_5element(X5element, pca5element, reds5element, blues5element, title = " fenetres temporelles s_t")


class VGG16Bottom(nn.Module):
    def __init__(self, original_model, N):
        super(VGG16Bottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-N])

    def forward(self, x):
        x = self.features(x)
        return x


class VGG16feat():
    def __init__(self, device):
        vgg16 = torch.hub.load('pytorch/vision:v0.4.2', 'vgg16', pretrained=True)
        self.device = device
        self.model = VGG16Bottom(vgg16, 1)
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extractfeat(self, input):
        return self.model((self.normalize(input).unsqueeze(0)).to(self.device).float())


class AlexBottom(nn.Module):
    def __init__(self, original_model, N):
        super(AlexBottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-N])

    def forward(self, x):
        x = self.features(x)
        return x


class Alexfeat():
    def __init__(self, device):
        alex = torch.hub.load('pytorch/vision:v0.4.2', 'alexnet', pretrained=True)
        alex.eval()
        self.device = device
        self.model = AlexBottom(alex, 2)
        self.model.to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extractfeat(self, input):
        return self.model(((self.normalize(input).unsqueeze(0)).to(self.device)).float())


class AlexBottom(nn.Module):
    def __init__(self, original_model, N):
        super(AlexBottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-N])

    def forward(self, x):
        x = self.features(x)
        return x


class Resfeat():
    def __init__(self, device):
        res = torch.hub.load('pytorch/vision:v0.4.2', 'resnet18', pretrained=True)
        self.device = device
        self.model = ResBottom(res, 2)
        self.model.to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extractfeat(self, input):
        return self.model(((self.normalize(input).unsqueeze(0)).to(self.device)).float())


class ResBottom(nn.Module):
    def __init__(self, original_model, N):
        super(ResBottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-N])

    def forward(self, x):
        x = self.features(x)
        return x





print("Load model")
the_model = VUnet()
the_model.to(device)
the_model.load_state_dict(torch.load(config.path.result_path_model, map_location=torch.device('cpu')))
the_model.eval()

print("End model")

#x_latent,x11,x9,x7,x5,x3,x1 = the_model.encode(images.float().to(device))













modelvgg = Resfeat(device)

VGGX1 = np.zeros((np.shape(X1i)[0],512,8,8))
latentX1 = np.zeros((np.shape(X1i)[0],512,4,4))

for i in range(np.shape(X1i)[0]):
  VGGX1[i] = modelvgg.extractfeat((X1i[i,16:288-16,16:288-16,:]/255)).cpu().detach()
  latentX1[i] = the_model.encode(trans_all(X1i[i,16:288-16,16:288-16,:]/255).unsqueeze(0).float().to(device))[0].cpu().detach()





VGGX1 = VGGX1.reshape(len(Y1),-1)
pcaVGGX1 = decomposition.PCA(n_components=2)
pcaVGGX1.fit(VGGX1)
X_pcaVGGX1 = pcaVGGX1.transform(VGGX1)
redsVGGX1 = Y1 == 0
bluesVGGX1 = Y1 == 1
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
X_tsneVGGX1 = tsne.fit_transform(VGGX1)
reducer = umap.UMAP(random_state=0)
embeddingVGGX1 = reducer.fit_transform(VGGX1)

latentX1 = latentX1.reshape(len(Y1),-1)
pcalatentX1 = decomposition.PCA(n_components=2)
pcalatentX1.fit(latentX1)
X_pcalatentX1 = pcalatentX1.transform(latentX1)
redslatentX1 = Y1 == 0
blueslatentX1 = Y1 == 1
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
X_tsnelatentX1 = tsne.fit_transform(latentX1)
reducer = umap.UMAP(random_state=0)
embeddinglatentX1 = reducer.fit_transform(latentX1)

plot_pca(X_pca1, pca1, reds1, blues1, title = " fenetres temporelles s_t")

plot_pca(X_pcalatentX1, pcalatentX1, redslatentX1, blueslatentX1, title = " fenetres temporelles s_t")
plt.show()

# ----------------------------------------------------------------------
# t-SNE embedding of the digits dataset
print("Computing t-SNE embedding")
t0 = time()
plot_embedding(X_tsne1,reds1, blues1,
               "t-SNE embedding fenetres temporelles s_t (time %.2fs)" %
               (time() - t0))


print("Computing t-SNE embedding")
t0 = time()
plot_embedding(X_tsnelatentX1,redslatentX1, blueslatentX1,
               "t-SNE embedding fenetres temporelles s_t (time %.2fs)" %
               (time() - t0))
plt.show()

# ----------------------------------------------------------------------
# Umap embedding of the digits dataset
print("Computing UMAP embedding")
t0 = time()
plot_embedding(embedding1,reds1, blues1,
               "umap embedding fenetres temporelles s_t (time %.2fs)" %
               (time() - t0))



print("Computing UMAP embedding")
t0 = time()
plot_embedding(embeddinglatentX1,redslatentX1, blueslatentX1,
               "umap embedding fenetres temporelles s_t (time %.2fs)" %
               (time() - t0))
plt.show()













def matplotlib_imshow(img, plt, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def f(x):
    return x

def preprocessing(img_name, trans_all):
  # get some random training images
  image = io.imread(img_name)
  input = image[:,:288,:]
  normals = image[:,288:2*288,:]
  diffuse = image[:,2*288:3 * 288,:]
  roughness = image[:,3*288:4 * 288,:]
  specular = image[:,4*288:5 * 288,:]
  #label = np.concatenate((normals,diffuse,roughness,specular),axis = 2)
  if trans_all:
      input_t = trans_all(input)
      normals_t = trans_all(normals)
      sample = {'input': input_t, 'label': normals_t}
  else:
      sample = {'input': input, 'label': normals}
  images, labels = sample["input"].unsqueeze(0), sample["label"].unsqueeze(0)
  return images, labels

def visualisation_finale(images, labels, sortie_to_plot, plt):
  # get some random training images
  # create grid of images
  img_grid = torchvision.utils.make_grid(images)
  img_grid_labels = torchvision.utils.make_grid(labels)
  img_grid_sortie_to_plot = torchvision.utils.make_grid(sortie_to_plot)
  # show images
  plt.figure(figsize=(25, 25))  # create a plot figure
  #create the first of two panels and set current axis
  plt.subplot(1, 3, 1)
  matplotlib_imshow(img_grid, plt, one_channel=False)
  plt.subplot(1, 3, 2)
  matplotlib_imshow(img_grid_labels, plt, one_channel=False)
  plt.subplot(1, 3, 3)
  matplotlib_imshow(img_grid_sortie_to_plot.cpu().detach(), plt, one_channel=False)
  plt.show()


def pipeline_all(img_name, trans_all, the_model, plt):
  images, labels = preprocessing(img_name, trans_all)
  x_latent = the_model.encoder(images.float().to(device))
  sortie_to_plot = the_model.decoder(x_latent.float().to(device))
  visualisation_finale(images, labels, sortie_to_plot, plt)

'''
def pipeline_inter(dico, images, labels, the_model, plt):
  x_latent_liste = []
  for index in range(len(dico)):
    x_latent_liste.append(dico[index].result)
  x_latent = trans_all(np.array([x_latent_liste]))
  sortie_to_plot = the_model.decoder(x_latent.float().to(device))
  visualisation_finale(images, labels, sortie_to_plot, plt)
'''