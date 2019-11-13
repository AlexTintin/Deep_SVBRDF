import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, models, transforms
from utils.tools import *


def initVGG16(target, label, device):
    vgg16 = models.vgg16(pretrained=True)
    vgg16.to(device)
    vgg16.eval()
    return vgg16


def lossVGG16(model, target, label):
    trans_all = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    ttarget = trans_all(target)
    tlabel = trans_all(label)
    target_normal, target_diffuse, target_rough, target_spec = reconstruct_output(ttarget)
    label_normal, label_diffuse, label_rough, target_spec = reconstruct_output(label)
    out_target_normal = model(target_normal)
    out_label = model(tlabel)
    return torch.mean(torch.abs(out_label - out_target))