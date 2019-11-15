import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, models, transforms
from utils.tools import *

class VGG16loss():

    def __init__(self,device):
        vgg16 = models.vgg16(pretrained=True)
        self.device = device
        vgg16.to(device)
        vgg16.eval()
        for param in vgg16.parameters():
            param.requires_grad = False
        self.model = vgg16
        self.normalize = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def lossVGG16_l1(self, target, label):
        target_rec = reconstruct_output(target)
        label_rec = reconstruct_output(label)
        out_target = torch.zeros((4,1000))
        out_label = torch.zeros((4,1000))
        moy = torch.zeros((4,1000))
        for i in range(4):
            out_target[i] = self.model((self.normalize(target_rec[i].squeeze(0))).unsqueeze(0))
            out_label[i] = self.model((self.normalize(label_rec[i].squeeze(0))).unsqueeze(0))
            moy+= torch.abs(out_target[i]-out_label[i])
        return torch.mean(moy)
        '''
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
        '''
    def lossVGG16_rendering(self, target, label):
        out_label = self.model(self.normalize(label.squeeze(0)).unsqueeze(0).to(self.device))
        out_target = self.model(self.normalize(target.squeeze(0)).unsqueeze(0).to(self.device))
        return torch.mean(torch.abs(out_label - out_target))