import torch.nn as nn
import torch
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), 64, 256, 256)


class Selu(nn.Module):
    def forward(self, x):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * F.elu(x, alpha)

class unitglobalnetdown(nn.Module):
    def __init__(self, input_size, output_size,pad =0):
        super(unitglobalnetdown, self).__init__()

        self.IN = nn.BatchNorm2d(input_size)
        self.lrelu = nn.LeakyReLU(0.2,True)
        self.conv = nn.Conv2d(input_size, output_size, kernel_size=4,stride=2, padding = pad)
        self.FC1 = nn.Linear(input_size,input_size)
        self.FC2 = nn.Linear(2*input_size, output_size)
        self.flat = Flatten()
        self.unflat = UnFlatten()
        self.selu = Selu()

    def forward(self, xu, xg):
        x1u = self.IN(xu)
        x1g = self.FC1(self.flat(xg))
        x2u = x1u
        x3u = self.lrelu(x2u)
        x4u = self.conv(x3u)
        x2g = self.FC2(self.flat(torch.cat([torch.mean(torch.mean(x1u, dim=2),dim=2), xg], dim=1)))
        x3g = self.selu(x2g)

        return x4u, x3g

class unitglobalnetup(nn.Module):
    def __init__(self, input_size, output_size, pad = 0):
        super(unitglobalnetup, self).__init__()

        self.IN = nn.BatchNorm2d(input_size)
        self.lrelu = nn.LeakyReLU(0.2,True)
        self.conv = nn.Conv2d(input_size, int(input_size), kernel_size=4,padding = pad)
        self.FC1 = nn.Linear(output_size,output_size)
        self.FC2 = nn.Linear(output_size+input_size, output_size)
        self.flat = Flatten()
        self.unflat = UnFlatten()
        self.selu = Selu()

    def forward(self, xu, xg):
        x1u = self.IN(xu)
        x1g = self.FC1(self.flat(xg))
        x2u = x1u
        x3u = self.lrelu(x2u)
        x4u = self.conv(x3u)
        x2g = self.FC2(self.flat(torch.cat([torch.mean(torch.mean(x1u, dim=2),dim=2), xg], dim=1)))
        x3g = self.selu(x2g)

        return x4u, x3g



class Unetglobal(nn.Module):
    def __init__(self):
        super(Unetglobal, self).__init__()

        #self.inc = nn.Conv2d(3, 64, kernel_size=4, stride=2)
        self.inc = unitglobalnetdown(3, 64)
        self.down1 = unitglobalnetdown(64, 128)
        self.down2 = unitglobalnetdown(128, 256)
        self.down3 = unitglobalnetdown(256, 512)
        self.down4 = unitglobalnetdown(512, 512)
        self.down5 = unitglobalnetdown(512, 512)
        self.down6 = unitglobalnetdown(512, 512,1)
        self.up1 = unitglobalnetup(2*512, 512,1)
        self.up2 = unitglobalnetup(2 * 512, 512,4)
        self.up3 = unitglobalnetup(2 * 512, 512,2)
        self.up4 = unitglobalnetup(2*512, 256)
        self.up5 = unitglobalnetup(2*256, 128)
        self.up6 = unitglobalnetup(2*128, 64)
        self.outc = nn.Conv2d(64*2, 3, kernel_size=4)
        self.drop = nn.Dropout2d(0.5)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.unmawpool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)



    def encoder(self,x,y):
        x1,y = self.inc(x,y)
        x2,y1 = self.down1(x1,y)
        x3,y2 = self.down2(x2,y1)
        x4,y3 = self.down3(x3,y2)
        x5,y4 = self.down4(x4,y3)
        x6,y5 = self.down5(x5,y4)
        x7,y6 = self.down6(x6,y5)
        return x7,x6,x5,x4,x3,x2,x1,y6


    def decoder(self,x7,x6,x5,x4,x3,x2,x1,y6):
        x10 = self.unmawpool(x7)
        x11,y7 = self.up1(torch.cat([x10, x10],dim=1),y6)
        x12 = self.unmawpool(x11)
        x13,y8 = self.up2(torch.cat([x12, x6], dim=1), y7)
        x14 = self.unmawpool(x13)
        x15,y9 = self.up3(torch.cat([x13, x5], dim=1), y8)
        x16 = self.unmawpool(x15)
        x17,y10 = self.up4(torch.cat([x16, x4], dim=1), y9)
        x18 = self.unmawpool(x17)
        x19,y11 = self.up5(torch.cat([x18, x3], dim=1), y10)
        x20 = self.unmawpool(x19)
        x21,y12 = self.up6(torch.cat([x20, x2], dim=1), y11)
        x22 = self.outc(torch.cat([x21, x1]))

        return x22

    def forward(self, x, y):
        xlatent, x6,x5,x4,x3,x2,x1,y6 =  self.encoder(x, y)
        x = self.decoder( xlatent, x6, x5, x4, x3, x2, x1, y6)
        return x