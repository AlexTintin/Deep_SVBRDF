import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, models, transforms
from utils.tools import *
from utils import config


class VGG16Bottom(nn.Module):
    def __init__(self, original_model,N):
        super(VGG16Bottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-N])

    def forward(self, x):
        x = self.features(x)
        return x


class VGG16loss():

    def __init__(self,device):
        vgg16 = models.vgg16(pretrained=True)
        self.device = device
        self.model = VGG16Bottom(vgg16,4)
        self.model.to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.normalize = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    def lossVGG16_l1(self, target, label):
        target_rec = reconstruct_output(target)
        label_rec = reconstruct_output(label)
        out_target = torch.zeros((4,3,256,256))
        out_label = torch.zeros((4,3,256,256))
        moy = torch.zeros((4,3,256,256))
        for i in range(4):
            out_target[i] = self.model((self.normalize(target_rec[i].squeeze(0))).unsqueeze(0))
            out_label[i] = self.model((self.normalize(label_rec[i].squeeze(0))).unsqueeze(0))
            moy+= torch.abs(out_target[i]-out_label[i])
        return torch.mean(moy)


    def lossVGG16_l1test(self, target, label):
        out_target = torch.zeros((2, 3, 256, 256))
        out_label = torch.zeros((2, 3, 256, 256))
        moy = torch.zeros((3, 256, 256))
        for i in range(1):
            out_target[i] = self.model((self.normalize(target[i].squeeze(0))).unsqueeze(0))
            out_label[i] = self.model((self.normalize(label[i].squeeze(0))).unsqueeze(0))
            moy += torch.abs(out_target[i]-out_label[i])
        return torch.mean(moy)

    def lossVGG16_rendering(self, target, label):
        loglabel = (torch.log(label+0.01)-np.log(0.01))/(np.log(1.01)-np.log(0.01))
        logtarget = (torch.log(target + 0.01) - np.log(0.01)) / (np.log(1.01) - np.log(0.01))
        out_label = self.model(self.normalize(loglabel.squeeze(0)).unsqueeze(0))
        out_target = self.model(self.normalize(logtarget.squeeze(0)).unsqueeze(0))
        return torch.mean(torch.abs(out_label - out_target))