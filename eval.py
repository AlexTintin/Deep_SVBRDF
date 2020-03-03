#from utils import config
import glob, os
import torch
import matplotlib.pyplot as plt
from src import dataloader
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from src.Model_Unet import *
from src.VAE import *
from src.DisentAE import *
from src.AE_Goa import *
from src.UnetDisentagled import *
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
from src.Pretrain_encode import *
import matplotlib.image as mpimg
import argparse
import scipy.misc
from PIL import Image




parser = argparse.ArgumentParser()
parser.add_argument("--data_path_realtrain",type=str, default="../DeepMaterialsData/trainBlended")
parser.add_argument("--data_path_train",type=str, default="../trainon2")
parser.add_argument("--data_path_val",type=str, default="../valon1")
parser.add_argument("--data_path_test",type=str, default="./../my_little_dataset/test")
parser.add_argument("--result_path_model",type=str, default="../../Deep_SVBRDF_local/content/mytraining_DUNET_VAE_l1_2mat_specbis2.pt" )
parser.add_argument("--logs_tensorboard",type=str, default="../../runs/testDUNET_Resnet4")
parser.add_argument("--load_path", type=str, default="../../Deep_SVBRDF_local/content/mytraining_DUNET_VAE_l1_2mat_spec.pt")
parser.add_argument("--seed", type=int, default=0 )
parser.add_argument("--num_epochs", type=int, default=10000)
parser.add_argument("--learning_rate", type=float, default=0.000005)
parser.add_argument("--weight_decay", type=float, default=0.000000001 )
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--trainset_division", type=int, default=1000, help="scale images to this size before cropping to 256x256")
parser.add_argument("--real_training", type=bool, default=False)
parser.add_argument("--loss", type=str, default="deep")

config = parser.parse_args()



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#config = config()
writer = SummaryWriter(config.logs_tensorboard)
print()
print("Use Hardware : ", device)
#Reproductibilites
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#Load data
print()
print("Load data")


trans_all = transforms.Compose([
        transforms.ToTensor()
    ])


dataload_test = dataloader.Dataloader(config, phase = "test", transform=trans_all)
dataloadered_test = DataLoader(dataload_test, batch_size=config.batch_size,
                        shuffle=True, num_workers=config.num_workers)

dataload_val = dataloader.Dataloader(config, phase = "val", transform=trans_all)
dataloadered_val = DataLoader(dataload_val, batch_size=config.batch_size,
                        shuffle=True, num_workers=config.num_workers)

dataload_train = dataloader.Dataloader(config, phase = "train", period = 'eval', transform=trans_all) #67 bizarre #37 pas mal
dataloadered_train = DataLoader(dataload_train, batch_size=config.batch_size,
                        shuffle=True, num_workers=config.num_workers)


dataloaders = {'train': dataloadered_train}#, 'val': dataloadered_val, 'test':dataloadered_test}


# get some random training images
dataiter = iter(dataloadered_train)
sample = dataiter.next()
sample = dataiter.next()
sample = dataiter.next()

'''
dataiter = iter(dataloadered_val)
sample2 = dataiter.next()
sample2 = dataiter.next()
sample2 = dataiter.next()
sample2 = dataiter.next()
'''
#images, labels = sample["input"], sample["label"]
imagesx, imagesy, labels = sample["inputx"],sample["inputy"], sample["label"]
#imagesx2, imagesy2, labels2 = sample2["inputx"],sample2["inputy"], sample2["label"]
print("End Load data")
print()
print("Load model")
the_model = DUnet()
the_model.to(device)
#writer.add_graph(the_model, imagesx.float().to(device),imagesy.float().to(device))
#writer.close()
#the_model.load_state_dict(torch.load(config.path.load_path+ 'l1_15000', map_location=torch.device('cpu')))

state_dict = torch.load(config.result_path_model)
# create new OrderedDict that does not contain `module.`
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
# load params
the_model.load_state_dict(new_state_dict)

#the_model.load_state_dict(torch.load(config.result_path_model, map_location=torch.device('cpu')))
the_model.eval()

print("End model")


#sortie_to_plot = the_model(imagesx.float().to(device),imagesy2.float().to(device))
sortie_to_plot2 = the_model(imagesx.float().to(device),imagesy.float().to(device))

# create grid of images
#img_grid = torchvision.utils.make_grid(imagesy2)#(np.log(images+0.01)-np.log(0.01))/(np.log(1.01)-np.log(0.01))
#img_grid_sortie_to_plot_normals =torchvision.utils.make_grid((sortie_to_plot[:,0:3,:,:].cpu().detach()))
img_grid_labels = torchvision.utils.make_grid(imagesx)
img_grid_labelsa = torchvision.utils.make_grid(imagesy)
img_grid_labels2 = torchvision.utils.make_grid(labels)
#img_grid_labels2 = torchvision.utils.make_grid(labels[:,0:3,:,:])
#img_grid_labels3 = torchvision.utils.make_grid((labels[:,3:4,:,:]))
#img_grid_sortie_to_plot2 = torchvision.utils.make_grid((labels[:,4:7,:,:]))

matplotlib_imshow(img_grid_labels.cpu().detach(), one_channel=False)
plt.show()

matplotlib_imshow(img_grid_labelsa.cpu().detach(), one_channel=False)
plt.show()


matplotlib_imshow(img_grid_labels2.cpu().detach(), one_channel=False)
plt.show()

#matplotlib_imshow(img_grid_labels3.cpu().detach(), one_channel=False)
#plt.show()
#matplotlib_imshow(img_grid_sortie_to_plot2.cpu().detach(), one_channel=False)
#plt.show()

img_grid_labels2 = torchvision.utils.make_grid(sortie_to_plot2)
#img_grid_labels2 = torchvision.utils.make_grid(sortie_to_plot2[:,0:3,:,:])
#mg_grid_sortie_to_plot2 = torchvision.utils.make_grid((sortie_to_plot[:,0:3,:,:]))
#img_grid_labels3 = torchvision.utils.make_grid((sortie_to_plot2[:,3:4,:,:]))
#img_grid_sortie_to_plot3 = torchvision.utils.make_grid((sortie_to_plot[:,3:6,:,:]))
#img_grid_labels4 = torchvision.utils.make_grid((sortie_to_plot2[:,4:7,:,:]))
#img_grid_sortie_to_plot4 = torchvision.utils.make_grid((sortie_to_plot[:,6:9,:,:]))

#matplotlib_imshow(img_grid, one_channel=False)
#plt.show()
#matplotlib_imshow(img_grid_sortie_to_plot_normals.cpu().detach(), one_channel=False)
#plt.show()
matplotlib_imshow(img_grid_labels2.cpu().detach(), one_channel=False)
plt.show()

#matplotlib_imshow(img_grid_sortie_to_plot2.cpu().detach(), one_channel=False)
#plt.show()
#matplotlib_imshow(img_grid_labels3.cpu().detach(), one_channel=False)
#plt.show()

#matplotlib_imshow(img_grid_labels4.cpu().detach(), one_channel=False)
#plt.show()



'''
list_light, list_view = get_wlvs_np(256, 10)
viewlight = list_light[9]
#B = render(torch.cat([imagesy2.float().to(device), sortie_to_plot], dim=1), viewlight[1], viewlight[0],roughness_factor=0.0)
A = render(torch.cat([imagesy.float().to(device), sortie_to_plot2], dim=1), viewlight[1], viewlight[0],roughness_factor=0.0)
Al = render(torch.cat([imagesy.float().to(device), labels.float().to(device)], dim=1), viewlight[1], viewlight[0],roughness_factor=0.0)


im = A.mean(dim=0)
img = im.detach().numpy()
img = np.transpose(img, (1, 2, 0))
plt.imshow(img)
plt.show()

matplotlib_imshow(torchvision.utils.make_grid(A.detach()), one_channel=False)
plt.show()
#matplotlib_imshow(torchvision.utils.make_grid(B.detach()), one_channel=False)
#plt.show()
matplotlib_imshow(torchvision.utils.make_grid(Al.detach()), one_channel=False)
plt.show()
'''
'''
for i in range(4):
    #img_grid = torchvision.utils.make_grid(images)
    #matplotlib_imshow(img_grid, one_channel=False)
    #plt.show()
    #images += 0.1*torch.Tensor(images.size()).uniform_(0, 1)
    

    #play with latent space


    xlat,i = torch.max(x_latent.squeeze(0),axis=1)
    nplat = xlat.cpu().detach().numpy()
    tsne(nplat,i)
plt.show()
'''
'''
M = 100000
Nbest=np.zeros(4)

A =labels[:,3:6,:,:]#sortie_to_plot[:,3:6,:,:].cpu().detach()



#print(np.shape(n.squeeze(0)))


list_light, list_view = get_wlvs_np(256, 10)
viewlight = list_light[9]
rendered = render(labels.float().to(device), viewlight[1], viewlight[0], roughness_factor=0.0)
#save_image(images.float().to(device),labels.float().to(device),rendered.float().to(device),'../label',True)


x_latent,x11,x9,x7,x5,x3,x1 = the_model.encode(images.float().to(device))
sortie_to_plot = the_model.decode(x_latent.float().to(device),x11.float().to(device),x9.float().to(device),
                                  x7.float().to(device),x5.float().to(device),x3.float().to(device),
                                  x1.float().to(device))
#imageio.imsave("img2.png", np.transpose(images.squeeze(0), (1, 2, 0)))
#plt.savefig('im.png')
rendered = render(sortie_to_plot, viewlight[1], viewlight[0], roughness_factor=0.0)
save_image(images.float().to(device),sortie_to_plot,rendered.float().to(device),'../'+config.train.loss,True)

writer.add_image('4_images_of_dataset_input', torchvision.utils.make_grid(images))
writer.add_image('4_images_of_dataset_output', torchvision.utils.make_grid(labels[:,:3,:,:]))
writer.add_image('4_images_of_model_output', torchvision.utils.make_grid(sortie_to_plot[:,:3,:,:]))
'''
'''
for a in range(4):
    M = 100000
    if a==0:
        A = labels[:,:3,:,:]
    elif a==1:
        A = labels[:, 3:6, :, :]
    elif a==3:
        A = labels[:, 7:, :, :]
    else:
        A= labels[:,6:7,:,:]

    for N in range(502):

        x_latent_new = torch.cat([torch.ones(1,N,4,4),torch.zeros(1,10,4,4),torch.ones(1,512-N-10,4,4)],dim=1)

        print(N)


        #x_latent+=torch.Tensor(2, 512,4,4).uniform_(0, 1).to(device)
        sortie_to_plotnew = the_model.decode(x_latent_new.float().to(device),x11.float().to(device),x9.float().to(device),
                                      x7.float().to(device),x5.float().to(device),x3.float().to(device),
                                      x1.float().to(device))


        #sortie_to_plot = the_model(images.float().to(device))
        if a == 0:
            b = sortie_to_plotnew[:, :3, :, :].cpu().detach()
        elif a == 1:
            b = sortie_to_plotnew[:, 3:6, :, :].cpu().detach()
        elif a == 3:
            b = sortie_to_plotnew[:, 7:, :, :].cpu().detach()
        else:
            b = sortie_to_plotnew[:, 6:7, :, :].cpu().detach()

        m = abs((A-b).sum())
        if m<M:
            Nbest[a] = N
            M=m

print(Nbest)
'''
'''
x_latent_new = torch.cat([torch.ones(1,397,4,4),torch.zeros(1,23,4,4),torch.ones(1,512-420,4,4)],dim=1)

sortie_to_plotnew = the_model.decode(x_latent_new.float().to(device),x11.float().to(device),x9.float().to(device),
                                      x7.float().to(device),x5.float().to(device),x3.float().to(device),
                                      x1.float().to(device))

rendered = render(sortie_to_plotnew, viewlight[1], viewlight[0], roughness_factor=0.0)
save_image(images.float().to(device),sortie_to_plotnew,rendered.float().to(device),'A')
'''
'''
# create grid of images
img_grid = torchvision.utils.make_grid(images)#(np.log(images+0.01)-np.log(0.01))/(np.log(1.01)-np.log(0.01))
#img_grid_labels_normals = torchvision.utils.make_grid(deprocess(labels[:,:3,:,:]))#torchvision.utils.make_grid(torch.cat([labels[:,:2,:,:],torch.ones((config.train.batch_size,1,256,256))],dim=1))
img_grid_sortie_to_plot_normals =torchvision.utils.make_grid((sortie_to_plot[:,:3,:,:].cpu().detach()))

#img_grid_labels2 = torchvision.utils.make_grid((labels[:,3:6,:,:]))
img_grid_sortie_to_plot2 = torchvision.utils.make_grid((sortie_to_plot[:,3:6,:,:]))
#img_grid_labels3 = torchvision.utils.make_grid((torch.cat([labels[:,6:7,:,:],labels[:,6:7,:,:],labels[:,6:7,:,:]],dim=1)))
img_grid_sortie_to_plot3 = torchvision.utils.make_grid((torch.cat([sortie_to_plot[:,6:7,:,:],sortie_to_plot[:,6:7,:,:],sortie_to_plot[:,6:7,:,:]],dim=1)))
#img_grid_labels4 = torchvision.utils.make_grid((labels[:,7:,:,:]))
img_grid_sortie_to_plot4 = torchvision.utils.make_grid((sortie_to_plot[:,7:,:,:]))


img_grid = torchvision.utils.make_grid(images)#(np.log(images+0.01)-np.log(0.01))/(np.log(1.01)-np.log(0.01))
img_grid_labels_1 = torchvision.utils.make_grid((labels[:,:3,:,:]))#torchvision.utils.make_grid(torch.cat([labels[:,:2,:,:],torch.ones((config.train.batch_size,1,256,256))],dim=1))
img_grid_sortie_to_plot_normals_new =torchvision.utils.make_grid((sortie_to_plotnew[:,:3,:,:].cpu().detach()))

img_grid_labels2 = torchvision.utils.make_grid((labels[:,3:6,:,:]))
img_grid_sortie_to_plot2_new = torchvision.utils.make_grid((sortie_to_plotnew[:,3:6,:,:]))
img_grid_labels3 = torchvision.utils.make_grid((torch.cat([labels[:,6:7,:,:],labels[:,6:7,:,:],labels[:,6:7,:,:]],dim=1)))
img_grid_sortie_to_plot3_new = torchvision.utils.make_grid((torch.cat([sortie_to_plotnew[:,6:7,:,:],sortie_to_plotnew[:,6:7,:,:],sortie_to_plotnew[:,6:7,:,:]],dim=1)))
img_grid_labels4 = torchvision.utils.make_grid((labels[:,7:,:,:]))
img_grid_sortie_to_plot4_new = torchvision.utils.make_grid((sortie_to_plotnew[:,7:,:,:]))


# show images
matplotlib_imshow(img_grid, one_channel=False)
plt.show()
matplotlib_imshow(img_grid_sortie_to_plot_normals_new.cpu().detach(), one_channel=False)
plt.show()
matplotlib_imshow(img_grid_sortie_to_plot_normals.cpu().detach(), one_channel=False)
plt.show()
matplotlib_imshow(img_grid_sortie_to_plot2_new.cpu().detach(), one_channel=False)
plt.show()
matplotlib_imshow(img_grid_sortie_to_plot2.cpu().detach(), one_channel=False)
plt.show()
matplotlib_imshow(img_grid_sortie_to_plot3_new.cpu().detach(), one_channel=False)
plt.show()
matplotlib_imshow(img_grid_sortie_to_plot3.cpu().detach(), one_channel=False)
plt.show()
matplotlib_imshow(img_grid_sortie_to_plot4_new.cpu().detach(), one_channel=False)
plt.show()
matplotlib_imshow(img_grid_sortie_to_plot4.cpu().detach(), one_channel=False)
plt.show()
matplotlib_imshow(img_grid_labels_1.cpu().detach(), one_channel=False)
plt.show()
matplotlib_imshow(img_grid_labels2.cpu().detach(), one_channel=False)
plt.show()
matplotlib_imshow(img_grid_labels3.cpu().detach(), one_channel=False)
plt.show()
matplotlib_imshow(img_grid_labels4.cpu().detach(), one_channel=False)
plt.show()
# write to tensorboard


'''