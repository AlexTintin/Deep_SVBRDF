import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, models, transforms
from utils.tools import *

class VGG16loss():

    def __init__(self,device):
        vgg16 = models.vgg16(pretrained=True)
        vgg16.to(device)
        vgg16.eval()
        for param in vgg16.parameters():
            param.requires_grad = False
        self.model = vgg16


    def lossVGG16(self, target, label):
        target_normal, target_diffuse, target_rough, target_spec = reconstruct_output(target)
        label_normal, label_diffuse, label_rough, label_spec = reconstruct_output(label)
        out_target_normal = self.model(target_normal)
        out_target_diffuse = self.model(target_diffuse)
        out_target_rough = self.model(target_rough)
        out_target_spec = self.model(target_spec)
        out_label_normal = self.model(label_normal)
        out_label_diffuse = self.model(label_diffuse)
        out_label_rough = self.model(label_rough)
        out_label_spec = self.model(label_spec)
        return torch.mean(torch.abs(out_target_normal - out_label_normal)
                          +torch.abs(out_target_diffuse - out_label_diffuse)
                          +torch.abs(out_target_rough - out_label_rough)
                          +torch.abs(out_target_spec - out_label_spec))