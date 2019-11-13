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
            nn.ReLU(True),
            nn.Conv2d(output_size, output_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_size),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.doubleconv(x)
        return x



class VUnet(nn.Module):
    def __init__(self):
        super(VUnet, self).__init__()

        self.inc = doubleConv(3, 64)
        self.down1 = doubleConv(64, 128)
        self.down2 = doubleConv(128, 256)
        self.down3 = doubleConv(256, 512)
        self.down4 = doubleConv(512, 512)
        self.down5 = doubleConv(512, 512)
        self.down6 = doubleConv(512, 512)
        self.up1 = doubleConv(512+512, 512)
        self.up2 = doubleConv(512 + 512, 512)
        self.up3 = doubleConv(512 + 512, 512)
        self.up4 = doubleConv(512+256, 256)
        self.up5 = doubleConv(256+128, 128)
        self.up6 = doubleConv(128+64, 64)
        self.outc = nn.Conv2d(64, 10, kernel_size=1)
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
        x10 = self.maxpool(x9)
        x11 = self.down5(x10)
        x12 = self.maxpool(x11)
        x13 = self.down6(x12)
        return x13,x11,x9,x7,x5,x3,x1

    def decode(self,x13,x11,x9,x7,x5,x3,x1):
        x10a = self.unmawpool(x13)
        x11a = self.up1(torch.cat([x10a, x11], dim=1))
        x10b = self.unmawpool(x11a)
        x11b = self.up2(torch.cat([x10b, x9], dim=1))
        x10 = self.unmawpool(x11b)
        x11 = self.up3(torch.cat([x10, x7], dim=1))
        x12 = self.unmawpool(x11)
        x13 = self.up4(torch.cat([x12, x5], dim=1))
        x14 = self.unmawpool(x13)
        x15 = self.up5(torch.cat([x14, x3], dim=1))
        x16 = self.unmawpool(x15)
        x17 = self.up6(torch.cat([x16, x1], dim=1))
        # x18 = self.unmawpool(x17)
        x19 = self.outc(x17)
        return self.sig(x19)

    def reparametrize(self,x):
        mu = self.flat(x)
        logvar = self.flat(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std, mu, logvar


    def forward(self, x):
        xlatent, x11,x9,x7,x5,x3,x1 = self.encode(x)
        #z,mu,logvar = self.reparametrize(xlatent)
        x = self.decode(xlatent, x11,x9,x7,x5,x3,x1)
        return x#, mu,logvar


    def loss_function(self, reco_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(reco_x, x, size_average=False)

                # see Appendix B from VAE paper:
                    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
                    # https://arxiv.org/abs/1312.6114
                    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD
