import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), 16, 280, 280)


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(6,16,kernel_size=5),
            nn.ReLU(True),
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
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
