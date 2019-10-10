import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), 3, 288, 288)


class AE_Linear(nn.Module):
    def __init__(self):
        super(AE_Linear, self).__init__()

        self.encoder = nn.Sequential(
            Flatten(),
            nn.Linear(288 * 288 * 3, 120),
            nn.ReLU(True),
            nn.Linear(120, 84),
            nn.ReLU(True),
            nn.Linear(84, 10))

        self.decoder = nn.Sequential(
            nn.Linear(10, 84),
            nn.ReLU(True),
            nn.Linear(84, 120),
            nn.ReLU(True),
            nn.Linear(120, 288 * 288 * 3),
            UnFlatten()
            )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x
