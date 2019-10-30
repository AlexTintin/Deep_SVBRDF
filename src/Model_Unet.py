import torch.nn as nn
import torch

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), 16, 280, 280)

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



class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        self.inc = doubleConv(3, 64)
        self.down1 = doubleConv(64, 128)
        self.down2 = doubleConv(128, 256)
        self.down3 = doubleConv(256, 512)
        self.down4 = doubleConv(512, 1024)
        self.up1 = doubleConv(1024+512, 512)
        self.up2 = doubleConv(512+256, 256)
        self.up3 = doubleConv(256+128, 128)
        self.up4 = doubleConv(128+64, 64)
        self.outc = nn.Conv2d(64, 3, kernel_size=1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.unmawpool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        '''     
        self.encoder = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(True),
            nn.Maxpool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(True),
            nn.Maxpool2d(2, 2),
            Flatten(),
            nn.Linear(280 * 280 * 16, 120),
            nn.ReLU(True),
            nn.Linear(120, 84),
            nn.ReLU(True),
            nn.Linear(84, 10))

        self.decoder = nn.Sequential(
            nn.Linear(10, 84),
            nn.ReLU(True),
            nn.Linear(84, 120),
            nn.ReLU(True),
            nn.Linear(120, 280 * 280 * 16),
            UnFlatten(),
            #nn.MaxUnpool2d(16, 16),
            nn.ConvTranspose2d(16,6,kernel_size=5),
            nn.ReLU(True),
            #nn.MaxUnpool2d(16, 16),
            nn.ConvTranspose2d(6,3,kernel_size=5),
            nn.ReLU(True),
            nn.Sigmoid())'''

    def forward(self, x):
        #start encoder
        x1 = self.inc(x)
        x2 = self.maxpool(x1)
        x3 = self.down1(x2)
        x4 = self.maxpool(x3)
        x5 = self.down2(x4)
        x6 = self.maxpool(x5)
        x7 = self.down3(x6)
        x8 = self.maxpool(x7)

        #latent space
        x9 = self.down4(x8)

        #decoder
        x10 = self.unmawpool(x9)
        x11 = self.up1(torch.cat([x10, x7], dim=1))
        x12 = self.unmawpool(x11)
        x13 = self.up2(torch.cat([x12, x5], dim=1))
        x14 = self.unmawpool(x13)
        x15 = self.up3(torch.cat([x14, x3], dim=1))
        x16 = self.unmawpool(x15)
        x17 = self.up4(torch.cat([x16, x1], dim=1))
        #x18 = self.unmawpool(x17)
        x19 = self.outc(x17)

        return x19