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
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(output_size, output_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_size),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x):
        x = self.doubleconv(x)
        return x


class DUnet(nn.Module):
    def __init__(self):
        super(DUnet, self).__init__()

        self.inc = doubleConv(3, 64)
        self.incvae = doubleConv(6, 64)
        self.down1 = doubleConv(64, 128)
        self.down2 = doubleConv(128, 256)
        self.down3 = doubleConv(256, 512)
        self.down4 = doubleConv(512, 512)
        self.down5 = doubleConv(512, 512)
        self.down6 = doubleConv(512, 512)
        self.down7 = doubleConv(512, 512)
        self.upm1 = doubleConv(512 + 512, 512)
        self.up0 = doubleConv(512 + 512, 512)
        self.up1 = doubleConv(512 + 512, 512)
        self.up2 = doubleConv(512 + 512, 512)
        self.up3 = doubleConv(512 + 512, 512)
        self.up4 = doubleConv(512 + 256, 256)
        self.up5 = doubleConv(256 + 128, 128)
        self.up6 = doubleConv(128 + 64, 64)
        self.outc = nn.Conv2d(64, 7, kernel_size=1)
        self.sig = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.unmawpool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.flat = Flatten()
        self.unflat = UnFlatten()
        # self.lin = nn.Linear(2,8192)

    def encodeUnet(self, x):
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
        x14 = self.maxpool(x13)
        x15 = self.down7(x14)
        return x15,x13, x11, x9, x7, x5, x3, x1

    def encodeVAE(self, x):
        x1 = self.incvae(x)
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
        x_appearence2 = self.down6(x12)
        x14 = self.maxpool(x_appearence2)
        x_appearence = self.down7(x14)
        return x_appearence2,x_appearence

    def decode(self, x_appearence2,x_appearence, x15,x13, x11, x9, x7, x5, x3, x1):
        xm1 = self.upm1(torch.cat([x_appearence, x15], dim=1))
        xm1a = self.unmawpool(xm1)
        x0 = self.up0(torch.cat([xm1a, x13], dim=1))
        x0 = self.up0(torch.cat([x_appearence2, x0], dim=1))
        x10a = self.unmawpool(x0)
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

    def reparametrize(self, x):
        mu = self.flat(x)
        logvar = self.flat(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std, mu, logvar

    def forward(self, x, y):
        xlatent,x13, x11, x9, x7, x5, x3, x1 = self.encodeUnet(y)
        x_appearence2,x_appearence = self.encodeVAE(torch.cat([x, y], dim=1))
        z = self.latent_sample(x_appearence)
        z2 = self.latent_sample(x_appearence2)
        x = self.decode(x_appearence2,x_appearence, xlatent,x13, x11, x9, x7, x5, x3, x1)
        return x,z,xlatent,z2,x13

    def latent_sample(slef, p):
        mean = p
        stddev = 1.0
        eps = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([stddev])).sample(p.size())
        return mean + eps.squeeze(-1).cuda()

    def latent_kl(self, q, p):
        mean1 = q
        mean2 = p
        kl = 0.5 * (mean2 - mean1) ** 2
        kl = torch.sum(kl, dim=[1, 2, 3])
        kl = torch.mean(kl)
        return kl

