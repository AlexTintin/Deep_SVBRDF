import torch.nn as nn
import torch

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
        return x9,x7,x5,x3,x1

    def decode(self,x,x7,x5,x3,x1):
        x10 = self.unmawpool(x)
        x11 = self.up1(torch.cat([x10, x7], dim=1))
        x12 = self.unmawpool(x11)
        x13 = self.up2(torch.cat([x12, x5], dim=1))
        x14 = self.unmawpool(x13)
        x15 = self.up3(torch.cat([x14, x3], dim=1))
        x16 = self.unmawpool(x15)
        x17 = self.up4(torch.cat([x16, x1], dim=1))
        # x18 = self.unmawpool(x17)
        x19 = self.outc(x17)
        return x19


    def forward(self, x):
        xlatent, x7,x5,x3,x1 = self.encode(x)
        x = self.decode(xlatent, x7,x5,x3,x1)
        return x


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