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



class DAE(nn.Module):
    def __init__(self):
        super(DAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2,True),
            nn.MaxPool2d(2, 2), # 128*128*32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2, 2),  # 64*64*64
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2, 2), # 32*32*128
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2, 2), # 16*16*256
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2, 2), # 8*8*512
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2, 2),  # 4*4*512
            nn.BatchNorm2d(512),


        )
            #Flatten(),
            #View((-1, 256 * 1 * 1))
            #nn.Linear(256, z_dim * 2)

        self.decoderN = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(1024*2, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),                                #512*2*2

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1),     #128*8*8
            nn.LeakyReLU(0.2, True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),      #64*16*16
            nn.LeakyReLU(0.2, True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),                                    #32*32*32

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),  # 16*64*64

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(16, 8, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),  # 8*128*128

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(8, 3, kernel_size=3, padding=1),

            nn.Tanh()
        )

        self.decoderD = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(1024*4, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),  # 512*2*2

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),

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
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),  # 16*64*64

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(16, 3, kernel_size=3, padding=1),

            nn.Tanh()
        )

        self.decoderR = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(1024 * 4, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),  # 512*2*2

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),

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
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),  # 16*64*64

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(16, 3, kernel_size=3, padding=1),
            #nn.LeakyReLU(0.2, True),  # 8*128*128

            nn.Tanh()
        )

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


            nn.Tanh()
        )

        self.decoderS = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(1024 * 4, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),  # 512*2*2

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),

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
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),  # 16*64*64

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(16, 3, kernel_size=3, padding=1),
            #nn.LeakyReLU(0.2, True),  # 8*128*128

            nn.Tanh()
        )

    def encode(self, x):
        return self.encoder(x)

    def decodeN(self, z):
        return self.decoderN(z)

    def decodeD(self, z):
        return self.decoderD(z)

    def decodeR(self, z):
        return self.decoderR(z)

    def decodeS(self, z):
        return self.decoderS(z)

    def decode(self, z):
        return self.decoder(z)

    def devide_latent(self,x_latent):
        x_normal = x_latent[:, :, 0, :]
        x_diffuse = x_latent[:, :, 1, :]
        x_roughness = x_latent[:, :, 2, :]
        x_specular = x_latent[:, :, 3, :]
        return x_normal, x_diffuse, x_roughness, x_specular


    def forward(self, x):
        x_latent = self.encode(x)
        x_normal, x_diffuse, x_roughness, x_specular = self.devide_latent(x_latent)
        imN = self.decodeN(x_normal.reshape(x_normal.size(0), 512 * 4, 1, 1))
        x_normNdiff = torch.cat([x_normal.reshape(x_normal.size(0), 512 * 4, 1, 1),
                                 x_diffuse.reshape(x_normal.size(0), 512 * 4, 1, 1)], dim=1)
        imD = self.decodeD(x_normNdiff)
        x_normNspec = torch.cat([x_normal.reshape(x_normal.size(0), 512 * 4, 1, 1),
                                 x_specular.reshape(x_normal.size(0), 512 * 4, 1, 1)], dim=1)
        imS = self.decodeS(x_normNspec)
        x_normNrough = torch.cat([x_normal.reshape(x_normal.size(0), 512 * 4, 1, 1),
                                  x_roughness.reshape(x_normal.size(0), 512 * 4, 1, 1)], dim=1)
        imR = self.decodeR(x_normNrough)
        return torch.cat([imN, imD, imR, imS], dim=1)
