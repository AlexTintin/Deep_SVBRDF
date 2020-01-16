import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, models, transforms

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), 512, 4, 4)

class doubleConv(nn.Module):
    def __init__(self, input_size, output_size):
        super(doubleConv, self).__init__()

        self.doubleconv = nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_size),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(output_size, output_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_size),
            nn.LeakyReLU(0.2,True),
        )

    def forward(self, x):
        x = self.doubleconv(x)
        return x



class DUnet(nn.Module):
    def __init__(self):
        super(DUnet, self).__init__()

        self.inc = doubleConv(3, 64)
        self.down1 = doubleConv(64, 128)
        self.down2 = doubleConv(128, 256)
        self.down3 = doubleConv(256, 512)
        self.down4 = doubleConv(512, 512)
        self.down5 = doubleConv(512, 512)
        self.down6 = doubleConv(512, 512)
        self.up0 = doubleConv(512 * 8, 512 * 4)
        self.up1 = doubleConv(512, 512)
        self.up2 = doubleConv(512, 512)
        self.up3 = doubleConv(512, 256)
        self.up4 = doubleConv(256, 128)
        self.up5 = doubleConv(128, 64)
        self.up6 = doubleConv(64, 32)
        self.up7 = doubleConv(32, 16)
        self.up8 = doubleConv(64, 16)
        self.outc = nn.Conv2d(16, 10, kernel_size=1)
        self.outcr = nn.Conv2d(16, 1, kernel_size=1)
        self.sig= nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.unmawpool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.flat = Flatten()
        self.unflat = UnFlatten()
        #self.lin = nn.Linear(2,8192)

    def encode(self, x):
        x1 = self.inc(x)
        x2 = self.maxpool(x1)
        x3 = self.down1(x2)
        x4 = self.maxpool(x3)
        x5 = self.down2(x4)
        x6 = self.maxpool(x5)
        x7 = self.down3(x6)
        x8 = self.maxpool(x7)
        x9 = self.down4(x8)
        x_latent = self.maxpool(x9)
        #x11 = self.down5(x10)
        #x12 = self.maxpool(x11)
        #x_latent = self.down6(x12)
        return x_latent

    def decodeD(self,x_latent):
        x = self.unmawpool(x_latent)
        x = self.up0(x)
        x = self.unmawpool(x)
        x = self.up1(x)
        x = self.unmawpool(x)
        x = self.up2(x)
        x = self.unmawpool(x)
        x = self.up3(x)
        x = self.unmawpool(x)
        x = self.up4(x)
        x = self.unmawpool(x)
        x = self.up5(x)
        x = self.unmawpool(x)
        x = self.up8(x)
        x = self.unmawpool(x)
        x = self.outc(x)
        return self.sig(x)

    def decodeS(self,x_latent):
        x = self.unmawpool(x_latent)
        x = self.up0(x)
        x = self.unmawpool(x)
        x = self.up1(x)
        x = self.unmawpool(x)
        x = self.up2(x)
        x = self.unmawpool(x)
        x = self.up3(x)
        x = self.unmawpool(x)
        x = self.up4(x)
        x = self.unmawpool(x)
        x = self.up5(x)
        x = self.unmawpool(x)
        x = self.up8(x)
        x = self.unmawpool(x)
        x = self.outc(x)
        return self.sig(x)

    def decodeR(self,x_latent):
        x = self.unmawpool(x_latent)
        x = self.up0(x)
        x = self.unmawpool(x)
        x = self.up1(x)
        x = self.unmawpool(x)
        x = self.up2(x)
        x = self.unmawpool(x)
        x = self.up3(x)
        x = self.unmawpool(x)
        x = self.up4(x)
        x = self.unmawpool(x)
        x = self.up5(x)
        x = self.unmawpool(x)
        x = self.up8(x)
        x = self.unmawpool(x)
        x = self.outcr(x)
        return self.sig(x)

    def decodeN(self,x_latent):
        x = self.unmawpool(x_latent)
        #x = self.up1(x)
        #x = self.unmawpool(x)
        x = self.up3(x)
        x = self.unmawpool(x)
        x = self.up4(x)
        x = self.unmawpool(x)
        x = self.up5(x)
        x = self.unmawpool(x)
        x = self.up8(x)
        x = self.unmawpool(x)
        x = self.outc(x)
        return self.sig(x)

    def devide_latent(self,x_latent):
        x_normal = x_latent[:, :, 0, :]
        x_diffuse = x_latent[:, :, 1, :]
        x_roughness = x_latent[:, :, 2, :]
        x_specular = x_latent[:, :, 3, :]
        return x_normal, x_diffuse, x_roughness, x_specular


    def forward(self, x):
        x_latent = self.encode(x)
        #print(x_latent.size())
        im = self.decodeN(x_latent)
        return im
    '''
    def forward(self, x):
        x_latent = self.encode(x)
        x_normal, x_diffuse, x_roughness, x_specular = self.devide_latent(x_latent)
        imN = self.decodeN(x_normal.reshape(x_normal.size(0),512*4,1,1))
        x_normNdiff = torch.cat([x_normal.reshape(x_normal.size(0), 512 * 4, 1, 1),
                                 x_diffuse.reshape(x_normal.size(0), 512 * 4, 1, 1)], dim=1)
        imD = self.decodeD(x_normNdiff)
        x_normNspec = torch.cat([x_normal.reshape(x_normal.size(0), 512 * 4, 1, 1),
                                 x_specular.reshape(x_normal.size(0), 512 * 4, 1, 1)], dim=1)
        imS = self.decodeS(x_normNspec)
        x_normNrough = torch.cat([x_normal.reshape(x_normal.size(0), 512 * 4, 1, 1),
                                 x_roughness.reshape(x_normal.size(0), 512 * 4, 1, 1)], dim=1)
        imR = self.decodeR(x_normNrough)
        return torch.cat([imN,imD,imR,imS],dim=1)
    '''