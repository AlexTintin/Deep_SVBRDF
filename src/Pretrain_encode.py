import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, models, transforms
from src.Pretrain_encode import *



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


class Resfeat():
    def __init__(self, device):
        res = torch.hub.load('pytorch/vision:v0.4.2', 'resnet18', pretrained=True)
        self.device = device
        self.model = ResBottom(res, 4)
        self.model.to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.normalize = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extractfeat(self, input):
        return self.model(self.normalize(input.squeeze(0)).unsqueeze(0))


class ResBottom(nn.Module):
    def __init__(self, original_model, N):
        super(ResBottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-N])

    def forward(self, x):
        x = self.features(x)
        return x




class DAEpretrained(nn.Module):
    def __init__(self,device):
        super(DAEpretrained, self).__init__()

        self.premodel = Resfeat(device)
        self.encoder = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2, 2),  # 16*16*256
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2, 2),  # 8*8*512
            nn.BatchNorm2d(512),

        )
            #Flatten(),
            #View((-1, 256 * 1 * 1))
            #nn.Linear(256, z_dim * 2)

        self.decoder = nn.Sequential(


            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1),  # 128*8*8
            nn.LeakyReLU(0.2, True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),  # 64*16*16
            nn.LeakyReLU(0.2, True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),  # 32*32*32

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(32, 3, kernel_size=3, padding=1),


            nn.Sigmoid()
        )

    def encode(self, x):
        x= self.premodel.extractfeat(x)
        x= self.encoder(x)
        return x

    def decode(self, z):
        return self.decoder(z)

    def devide_latent(self, x_latent):
        x_normal = x_latent[:, :, 0, 0]
        x_diffuse = x_latent[:, :, 1, 0]
        x_roughness = x_latent[:, :, 0, 1]
        x_specular = x_latent[:, :, 1, 1]
        return x_normal, x_diffuse, x_roughness, x_specular

    def forward(self, x):
        #print(x.size())
        x_latent = self.encode(x)
        #print(x_latent.size())
        im_N = self.decode(x_latent)
        # x_normal, x_diffuse, x_roughness, x_specular = self.devide_latent(x_latent)
        # im_N = self.decodeN(x_normal.view(x_normal.size(0),1024,1,1))
        # x_normNdiff = torch.cat([x_normal,x_diffuse],dim=1)
        # im_D = self.decodeD(x_normNdiff.view(x_normal.size(0),1024*2,1,1))
        # x_normNrough = torch.cat([x_normal, x_roughness], dim=1)
        # im_R = self.decodeR(x_normNrough.view(x_normal.size(0),1024*2,1,1))
        # x_normNspec = torch.cat([x_normal, x_specular], dim=1)
        # im_S = self.decodeS(x_normNspec.view(x_normal.size(0),1024*2,1,1))
        # output = torch.cat([im_N, im_D,im_R,im_S], dim=1)
        return im_N
