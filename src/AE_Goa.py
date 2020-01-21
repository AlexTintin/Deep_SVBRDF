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

class Conv(nn.Module):
    def __init__(self, input_size, output_size):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(input_size, output_size, kernel_size=4,padding=1, stride=(2, 2))

    def forward(self, x):
        x = self.conv(x)
        return torch.nn.init.normal_(self.conv.weight)



class AEG(nn.Module):
    def __init__(self):
        super(AEG, self).__init__()
        self.encoder0 = nn.Conv2d(12, 64, kernel_size=4,padding=1, stride=(2, 2))
        nn.init.normal_(self.encoder0.weight,0,0.02)
        self.encoder1 = nn.Conv2d(64, 128, kernel_size=4, padding=1, stride=(2, 2))
        torch.nn.init.normal_(self.encoder1.weight,0,0.02)
        self.encoder2 = nn.Conv2d(128, 256, kernel_size=4, padding=1, stride=(2, 2))
        torch.nn.init.normal_(self.encoder2.weight,0,0.02)
        self.encoder3 = nn.Conv2d(256, 512, kernel_size=4, padding=1, stride=(2, 2))
        torch.nn.init.normal_(self.encoder3.weight,0,0.02)
        self.encoder4 = nn.Conv2d(512, 512, kernel_size=4, padding=1, stride=(2, 2))
        torch.nn.init.normal_(self.encoder4.weight,0,0.02)
        self.lrelu = nn.LeakyReLU(0.2,True)
        self.batch = nn.BatchNorm2d(512)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.decoder0 = nn.ConvTranspose2d(512, 512, kernel_size=4, padding=1, stride=(2, 2))
        torch.nn.init.normal_(self.encoder0.weight, 0, 0.02)
        self.decoder1 = nn.ConvTranspose2d(512, 256, kernel_size=4, padding=1, stride=(2, 2))
        torch.nn.init.normal_(self.encoder1.weight, 0, 0.02)
        self.decoder2 = nn.ConvTranspose2d(256, 128, kernel_size=4, padding=1, stride=(2, 2))
        torch.nn.init.normal_(self.encoder2.weight, 0, 0.02)
        self.decoder3 = nn.ConvTranspose2d(128, 64, kernel_size=4, padding=1, stride=(2, 2))
        torch.nn.init.normal_(self.encoder3.weight, 0, 0.02)
        self.decoder4 = nn.ConvTranspose2d(64, 12, kernel_size=4, padding=1, stride=(2, 2))
        torch.nn.init.normal_(self.encoder4.weight, 0, 0.02)


    def encode(self, x):
        x = self.encoder0(x)
        x = self.lrelu(x)
        x = self.encoder1(x)
        x = self.lrelu(x)
        x = self.encoder2(x)
        x = self.lrelu(x)
        x = self.encoder3(x)
        x = self.lrelu(x)
        x = self.encoder4(x)
        x = self.batch(x)
        return x


    def decode(self, x):
        x = self.lrelu(x)
        x = self.decoder0(x)
        x = self.lrelu(x)
        x = self.decoder1(x)
        x = self.lrelu(x)
        x = self.decoder2(x)
        x = self.lrelu(x)
        x = self.decoder3(x)
        x = self.lrelu(x)
        x = self.decoder4(x)
        x = self.tanh(x)
        return x


    def forward(self, x):
        x_latent = self.encode(x)
        im = self.decode(x_latent)
        #x_normal, x_diffuse, x_roughness, x_specular = self.devide_latent(x_latent)
        #im_N = self.decodeN(x_normal.view(x_normal.size(0),1024,1,1))
        #x_normNdiff = torch.cat([x_normal,x_diffuse],dim=1)
        #im_D = self.decodeD(x_normNdiff.view(x_normal.size(0),1024*2,1,1))
        #x_normNrough = torch.cat([x_normal, x_roughness], dim=1)
        #im_R = self.decodeR(x_normNrough.view(x_normal.size(0),1024*2,1,1))
        #x_normNspec = torch.cat([x_normal, x_specular], dim=1)
        #im_S = self.decodeS(x_normNspec.view(x_normal.size(0),1024*2,1,1))
        #output = torch.cat([im_N, im_D,im_R,im_S], dim=1)
        return im