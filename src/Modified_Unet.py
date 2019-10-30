import torch.nn as nn
import torch
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), 16, 280, 280)


class selu(nn.Module):
    def forward(self, x):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * F.elu(x, alpha)

class unitglobalnet(nn.Module):
    def __init__(self, input_size, output_size):
        super(unitglobalnet, self).__init__()

        self.IN =  nn.BatchNorm2d(input_size)
        self.lrelu = nn.LeakyReLU(0.2,True)
        self.conv = nn.Conv2d(input_size, output_size, kernel_size=4, stride=2)
        self.FC1 = nn.Linear(input_size,output_size)
        self.FC2 = nn.Linear(output_size, output_size)

    def forward(self, xu, xg):
        x1u = self.IN(xu)
        x1g = self.FC1(xg)
        x2u = x1u+x1g
        x3u = self.lrelu(x2u)
        x4u = self.conv(x3u)
        x2g = self.FC2(torch.cat([torch.mean(x1u, dim=0), xg], dim=1))
        x3g = selu(x2g)

        return x4u, x3g



class Unetglobal(nn.Module):
    def __init__(self):
        super(Unetglobal, self).__init__()

        #self.inc = nn.Conv2d(3, 64, kernel_size=4, stride=2)
        self.inc = unitglobalnet(3, 64)
        self.down1 = unitglobalnet(64, 128)
        self.down2 = unitglobalnet(128, 256)
        self.down3 = unitglobalnet(256, 512)
        self.down4 = unitglobalnet(512, 512)
        self.down5 = unitglobalnet(512, 512)
        self.down6 = unitglobalnet(512, 512)
        self.up1 = unitglobalnet(2*512, 512)
        self.up2 = unitglobalnet(2 * 512, 512)
        self.up3 = unitglobalnet(2 * 512, 512)
        self.up4 = unitglobalnet(2*512, 256)
        self.up5 = unitglobalnet(2*256, 128)
        self.up6 = unitglobalnet(2*128, 64)
        self.outc = nn.Conv2d(64*2, 3, kernel_size=4)
        self.drop = nn.Dropout2d(0.5)
        #self.maxpool = nn.MaxPool2d(2, 2)
        self.unmawpool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


    def forward(self, x,y):
        #start encoder
        x1,y = self.inc(x,y)
        x2,y1 = self.down1(x1,y)
        x3,y2 = self.down2(x2,y1)
        x4,y3 = self.down3(x3,y2)
        x5,y4 = self.down4(x4,y3)
        x6,y5 = self.down5(x5,y4)
        x7,y6 = self.down6(x6,y5)


        #decoder
        x10 = self.unmawpool(x7)
        x11,y7 = self.up1(torch.cat([x10, x7], dim=1),y6)
        x12 = self.unmawpool(x11)
        x13,y8 = self.up2(torch.cat([x12, x6], dim=1), y7)
        x14 = self.unmawpool(x13)
        x15,y9 = self.up3(torch.cat([x14, x5], dim=1), y8)
        x16 = self.unmawpool(x15)
        x17,y10 = self.up4(torch.cat([x16, x4], dim=1), y9)
        x18 = self.unmawpool(x17)
        x19,y11 = self.up5(torch.cat([x18, x3], dim=1), y10)
        x20 = self.unmawpool(x19)
        x21,y12 = self.up6(torch.cat([x20, x2], dim=1), y11)
        x22 = self.outc(torch.cat([x21, x1]))

        return x22