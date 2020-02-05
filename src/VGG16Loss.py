import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, models, transforms
from utils.tools import *
#from utils import config


class VGG16Bottom(nn.Module):
    def __init__(self, original_model, N):
        super(VGG16Bottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-N])

    def forward(self, x):
        x = self.features(x)
        return x


class VGG16feat():
    def __init__(self, device):
        vgg16 = torch.hub.load('pytorch/vision:v0.4.2', 'vgg16', pretrained=True)
        self.device = device
        self.model = VGG16Bottom(vgg16, 1)
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extractfeat(self, input):
        return self.model((self.normalize(input).unsqueeze(0)).to(self.device).float())

    def VGG16Loss(self, target, label):
        target_rec = reconstruct_output(target)
        label_rec = reconstruct_output(label)
        t = 0
        for j in range(3):
            t += torch.mean(torch.abs(self.extractfeat(label_rec[j]) - self.extractfeat(target_rec[j])))
        return t



class VGG19Bottom(nn.Module):
    def __init__(self, original_model, N):
        super(VGG19Bottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-N])

    def forward(self, x):
        x = self.features(x)
        return x


class VGG19feat():
    def __init__(self, device):
        vgg19 = torch.hub.load('pytorch/vision:v0.4.2', 'vgg19', pretrained=True)
        self.device = device
        self.model = VGG19Bottom(vgg19, 1)
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extractfeat(self, input):
        return self.model((self.normalize(input).unsqueeze(0)).to(self.device).float())

    def VGG19Loss(self, target, label):
        target_rec = reconstruct_output(target)
        label_rec = reconstruct_output(label)
        t = 0
        for j in range(3):
            t+=torch.mean(torch.abs(self.extractfeat(label_rec[j])-self.extractfeat(target_rec[j])))
        return t


class AlexBottom(nn.Module):
    def __init__(self, original_model, N):
        super(AlexBottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-N])

    def forward(self, x):
        x = self.features(x)
        return x


class Alexfeat():
    def __init__(self, device):
        alex = torch.hub.load('pytorch/vision:v0.4.2', 'alexnet', pretrained=True)
        alex.eval()
        self.device = device
        self.model = AlexBottom(alex, 2)
        self.model.to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extractfeat(self, input):
        return self.model(((self.normalize(input).unsqueeze(0)).to(self.device)).float())

    def AlexLoss(self, target, label):
        target_rec = reconstruct_output(target)
        label_rec = reconstruct_output(label)
        t = 0
        for j in range(3):
            t+=torch.mean(torch.abs(self.extractfeat(label_rec[j])-self.extractfeat(target_rec[j])))
        return t


class Resfeat():
    def __init__(self, device):
        res = torch.hub.load('pytorch/vision:v0.4.2', 'resnet18', pretrained=True)
        self.device = device
        self.model = ResBottom(res, 4)
        self.model.to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.normalize = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extractfeat(self, input):
        return self.model(self.normalize(input.squeeze(0)).unsqueeze(0))

    def ResLoss(self, target, label):
        target_rec = reconstruct_output(target)
        label_rec = reconstruct_output(label)
        t = 0
        for j in range(3):
            t+=torch.mean(torch.abs(self.extractfeat(label_rec[j])-self.extractfeat(target_rec[j])))
        return t


class ResBottom(nn.Module):
    def __init__(self, original_model, N):
        super(ResBottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-N])

    def forward(self, x):
        x = self.features(x)
        return x


'''

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
        target_rec = reconstruct_output(target)
        label_rec = reconstruct_output(label)
        out_target = torch.zeros((3, 3, 256, 256))
        out_label = torch.zeros((3, 3, 256, 256))
        moy = torch.zeros((3, 3, 256, 256))
        for i in range(3):
            out_target[i] = self.model((self.normalize(target_rec[i].squeeze(0))).unsqueeze(0))
            out_label[i] = self.model((self.normalize(label_rec[i].squeeze(0))).unsqueeze(0))
            moy += torch.abs(out_target[i]-out_label[i])
        return torch.mean(moy)

    def lossVGG16_rendering(self, target, label):
        loglabel = (torch.log(label+0.01)-np.log(0.01))/(np.log(1.01)-np.log(0.01))
        logtarget = (torch.log(target + 0.01) - np.log(0.01)) / (np.log(1.01) - np.log(0.01))
        out_label = self.model(self.normalize(loglabel.squeeze(0)).unsqueeze(0))
        out_target = self.model(self.normalize(logtarget.squeeze(0)).unsqueeze(0))
        return torch.mean(torch.abs(out_label - out_target))

'''