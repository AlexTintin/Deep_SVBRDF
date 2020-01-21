from utils import config
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
from src.DisentAEmixUnet import *
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


trans_all = transforms.Compose([
        transforms.ToTensor()
    ])


dataload_test = dataloader.Dataloader(config, phase = "test", transform=trans_all)
dataloadered_test = DataLoader(dataload_test, batch_size=config.train.batch_size,
                        shuffle=True, num_workers=config.train.num_workers)

dataload_val = dataloader.Dataloader(config, phase = "val", transform=trans_all)
dataloadered_val = DataLoader(dataload_val, batch_size=config.train.batch_size,
                        shuffle=True, num_workers=config.train.num_workers)

dataload_train = dataloader.Dataloader(config, phase = "train", period = 'eval', transform=trans_all) #67 bizarre #37 pas mal
dataloadered_train = DataLoader(dataload_train, batch_size=config.train.batch_size,
                        shuffle=True, num_workers=config.train.num_workers)


dataloaders = {'train': dataloadered_train, 'val': dataloadered_val, 'test':dataloadered_test}


# get some random training images
dataiter = iter(dataloadered_train)
sample = dataiter.next()
images, labels = sample["input"], sample["label"]
print("End Load data")
print()
print("Load model")
the_model = DAEpretrained(device)
the_model.to(device)
writer.add_graph(the_model, images.float().to(device))
writer.close()
#the_model.load_state_dict(torch.load(config.path.load_path+ 'l1_15000', map_location=torch.device('cpu')))
the_model.load_state_dict(torch.load(config.path.result_path_model, map_location=torch.device('cpu')))
the_model.eval()

print("End model")


sortie_to_plot = the_model(images.float().to(device))

# create grid of images
img_grid = torchvision.utils.make_grid(images)#(np.log(images+0.01)-np.log(0.01))/(np.log(1.01)-np.log(0.01))
#img_grid_labels_normals = torchvision.utils.make_grid(deprocess(labels[:,:3,:,:]))#torchvision.utils.make_grid(torch.cat([labels[:,:2,:,:],torch.ones((config.train.batch_size,1,256,256))],dim=1))
img_grid_sortie_to_plot_normals =torchvision.utils.make_grid((sortie_to_plot[:,:3,:,:].cpu().detach()))

#img_grid_labels2 = torchvision.utils.make_grid((labels[:,3:6,:,:]))
#img_grid_sortie_to_plot2 = torchvision.utils.make_grid((sortie_to_plot[:,3:6,:,:]))
#img_grid_labels3 = torchvision.utils.make_grid((torch.cat([labels[:,6:7,:,:],labels[:,6:7,:,:],labels[:,6:7,:,:]],dim=1)))
#img_grid_sortie_to_plot3 = torchvision.utils.make_grid((torch.cat([sortie_to_plot[:,6:7,:,:],sortie_to_plot[:,6:7,:,:],sortie_to_plot[:,6:7,:,:]],dim=1)))
#img_grid_labels4 = torchvision.utils.make_grid((labels[:,7:,:,:]))
#img_grid_sortie_to_plot4 = torchvision.utils.make_grid((sortie_to_plot[:,7:,:,:]))

matplotlib_imshow(img_grid, one_channel=False)
plt.show()
matplotlib_imshow(img_grid_sortie_to_plot_normals.cpu().detach(), one_channel=False)
plt.show()




the_model = DAE()
the_model.to(device)


the_model.load_state_dict(torch.load(config.path.load_path, map_location=torch.device('cpu')))
the_model.eval()

print("End model")


sortie_to_plot = the_model(images.float().to(device))

# create grid of images
img_grid = torchvision.utils.make_grid(images)#(np.log(images+0.01)-np.log(0.01))/(np.log(1.01)-np.log(0.01))
#img_grid_labels_normals = torchvision.utils.make_grid(deprocess(labels[:,:3,:,:]))#torchvision.utils.make_grid(torch.cat([labels[:,:2,:,:],torch.ones((config.train.batch_size,1,256,256))],dim=1))
img_grid_sortie_to_plot_normals =torchvision.utils.make_grid((sortie_to_plot[:,:3,:,:].cpu().detach()))

#img_grid_labels2 = torchvision.utils.make_grid((labels[:,3:6,:,:]))
#img_grid_sortie_to_plot2 = torchvision.utils.make_grid((sortie_to_plot[:,3:6,:,:]))
#img_grid_labels3 = torchvision.utils.make_grid((torch.cat([labels[:,6:7,:,:],labels[:,6:7,:,:],labels[:,6:7,:,:]],dim=1)))
#img_grid_sortie_to_plot3 = torchvision.utils.make_grid((torch.cat([sortie_to_plot[:,6:7,:,:],sortie_to_plot[:,6:7,:,:],sortie_to_plot[:,6:7,:,:]],dim=1)))
#img_grid_labels4 = torchvision.utils.make_grid((labels[:,7:,:,:]))
#img_grid_sortie_to_plot4 = torchvision.utils.make_grid((sortie_to_plot[:,7:,:,:]))

matplotlib_imshow(img_grid_sortie_to_plot_normals.cpu().detach(), one_channel=False)
plt.show()
#matplotlib_imshow(img_grid_sortie_to_plot2.cpu().detach(), one_channel=False)
#plt.show()
#matplotlib_imshow(img_grid_sortie_to_plot3.cpu().detach(), one_channel=False)
#plt.show()
#matplotlib_imshow(img_grid_sortie_to_plot4.cpu().detach(), one_channel=False)
#plt.show()

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