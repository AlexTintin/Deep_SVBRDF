from __future__ import division

#https://github.com/msraig/DeepInverseRendering

import math
import random
import os

import numpy as np
import torch
from utils.tools import *
from torchvision import transforms, utils
import torchvision
import matplotlib.pyplot as plt

trans_all = transforms.Compose([
        transforms.ToTensor()
    ])

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def L1Loss(inputs, targets, weight = 1.0):
    diff = torch.abs(inputs - targets)
    return torch.mean(diff) * weight

def L1LogLoss(inputs, targets, weight = 1.0):
    return torch.mean(torch.abs(torch.log(inputs + 0.01) - torch.log(targets + 0.01))) * weight

def L2Loss(inputs, targets,  weight = 1.0):
    return torch.reduce_mean(torch.squared_difference(inputs, targets)) * weight

# diffuse, specular, roughness, normal,
def render(inputs, l, v,roughness_factor=0.0):
    INV_PI = 1.0 / math.pi
    EPS = 1e-12

    def GGX(NoH, roughness):
        alpha = roughness * roughness
        tmp = alpha / torch.clamp((NoH * NoH * (alpha * alpha - 1.0) + 1.0),min=1e-8)
        return tmp * tmp * INV_PI

    def SmithG(NoV, NoL, roughness):
        def _G1(NoM, k):
            return NoM / (NoM * (1.0 - k) + k)

        k = torch.clamp(roughness * roughness * 0.5,min=1e-8)
        return _G1(NoL, k) * _G1(NoV, k)

    def Fresnel(F0, VoH):
        coeff = VoH * (-5.55473 * VoH - 6.98316)
        return F0 + (1.0 - F0) * (2.0** coeff)


    #normal, diffuse, roughness, specular = torch.split(inputs, 4, axis=-1)
    normal = inputs[:, :3, :, :]
    diffuse = inputs[:, 3:6, :, :]
    roughness = torch.cat([inputs[:,6:7,:,:],inputs[:,6:7,:,:],inputs[:,6:7,:,:]],dim=1)
    specular = inputs[:, 7:, :, :]

   # img_grid_labels4 = torchvision.utils.make_grid(specular)
    #matplotlib_imshow(img_grid_labels4.cpu().detach(), one_channel=False)
    #plt.show()

    normal_size = (normal.size())
    res = torch.ones((normal_size[0],normal_size[1],normal_size[2],normal_size[3]))
    #l1 = np.ones((3, 256, 256))
   # v1 = np.ones((3, 256, 256))

    #v1 = trans_all(v)
    #l1 = trans_all(l)

    v1 = torch.cuda.FloatTensor(v)
    l1 = torch.cuda.FloatTensor(l)


    v1 = v1.permute(2, 0, 1)
    l1 = l1.permute(2, 0, 1)

    h = tensor_norm((l1 + v1) * 0.5)

    for i in range(normal_size[0]):

        n = tensor_norm(normal[i,:,:,:])
        s = deprocess(specular[i,:,:,:])
        d = deprocess(diffuse[i,:,:,:])
        r = deprocess(roughness[i,:,:,:])

        NoV = tensor_dot(n, v1)
        NoH = tensor_dot(n, h)
        NoL = tensor_dot(n, l1)
        VoH = tensor_dot(v1, h)

        NoH = torch.clamp(NoH,min=1e-8)
        NoV = torch.clamp(NoV,min=1e-8)
        NoL = torch.clamp(NoL, min=1e-8)
        VoH = torch.clamp(VoH,min=1e-8)

        f_d = d * INV_PI

        D = GGX(NoH, r)
        G = SmithG(NoV, NoL, r)
        F = Fresnel(s, VoH)
        f_s = D * G * F / (4.0 * NoL * NoV + EPS)
        res[i,:,:,:] = (f_d + f_s) * NoL * math.pi
        #img_grid_labels4 = torchvision.utils.make_grid(res[i,:,:,:])
        #matplotlib_imshow(img_grid_labels4.cpu().detach(), one_channel=False)
        #plt.show()

    return res


    # diffuse, specular, roughness, normal

def render_np(n,d,r,s, l, v, roughness_factor=0.0):
    INV_PI = 1.0 / math.pi
    EPS = 1e-12

    def GGX(NoH, roughness):
        alpha = roughness * roughness
        tmp = alpha / np.maximum(1e-8, (NoH * NoH * (alpha * alpha - 1.0) + 1.0))
        return tmp * tmp * INV_PI

    def SmithG(NoV, NoL, roughness):
        def _G1(NoM, k):
            return NoM / (NoM * (1.0 - k) + k)

        k = np.maximum(1e-8, roughness * roughness * 0.5)
        return _G1(NoL, k) * _G1(NoV, k)

    def Fresnel(F0, VoH):
        coeff = VoH * (-5.55473 * VoH - 6.98316)
        return F0 + (1.0 - F0) * np.power(2.0, coeff)


    n = numpy_norm(n)
    l = np.array(l)
    v = np.array(v)
    h = numpy_norm((l + v) * 0.5)

    NoH = numpy_dot(n, h)
    NoV = numpy_dot(n, v)
    NoL = numpy_dot(n, l)
    VoH = numpy_dot(v, h)

    NoH = np.maximum(NoH, 1e-8)
    NoV = np.maximum(NoV, 1e-8)
    NoL = np.maximum(NoL, 1e-8)
    VoH = np.maximum(VoH, 1e-8)

    f_d = d * INV_PI

    D = GGX(NoH, r)
    G = SmithG(NoV, NoL, r)
    F = Fresnel(s, VoH)
    f_s = D * G * F / (4.0 * NoL * NoV + EPS)

    res = (f_d + f_s) * NoL * math.pi

    return res


def log_view_light_dirs(light_camera_pos, N, Ns, Nd, output_folder, d_name='view_light.txt'):
    def _join(lst):
        if isinstance(lst, list):
            return ",".join([str(i) for i in lst])
        elif isinstance(lst, np.ndarray):
            return ",".join([str(i) for i in lst.tolist()])

    # inira
    name = os.path.join(output_folder, d_name)
    if N == -1:
        with open(name, 'w+') as f:
            f.write("inira Ns:%d, Nd%d\n" % (Ns, Nd))
            for item in light_camera_pos:
                light_pos, camera_pos = item

                f.write(_join(light_pos) + "\t" + _join(camera_pos) + '\n')

    # predefined views:
    else:
        with open(name, 'w+') as f:
            f.write("predefined N: %d\n" % N)
            for item in light_camera_pos:
                light_pos, camera_pos = item
                f.write(_join(light_pos) + "\t" + _join(camera_pos) + '\n')


def get_wlvs_np(scale_size, total_num=10):
    def generate(camera_pos_world):
        light_pos_world = camera_pos_world
        x_range = np.linspace(-1, 1, scale_size)
        y_range = np.linspace(-1, 1, scale_size)
        x_mat, y_mat = np.meshgrid(x_range, y_range)
        pos = np.stack([x_mat, -y_mat, np.zeros(x_mat.shape)], axis=-1)

        view_dir_world = numpy_norm(camera_pos_world - pos)
        light_dir_world = numpy_norm(light_pos_world - pos)

        light_dir_world = light_dir_world.astype(np.float32)
        view_dir_world = view_dir_world.astype(np.float32)
        return view_dir_world, light_dir_world

    def random_pos():
        x = random.uniform(-1.2, 1.2)
        y = random.uniform(-1.2, 1.2)
        # z = random.uniform(2.0, 4.0)
        z = 2.146
        return [x, y, z]

    def record(camera_pos):
        Wlvs.append(generate(camera_pos))
        camera_light_pos.append([camera_pos, camera_pos])

    Wlvs = []
    camera_light_pos = []

    record([0, 0, 2.146])

    current_count = len(Wlvs)

    if total_num > current_count:
        for i in range(total_num - current_count):
            pos = random_pos()
            record(pos)

    return Wlvs[:total_num], camera_light_pos[:total_num]


def recover_wlvs_np(filename, scale_size):
    def generate(camera_pos_world):
        light_pos_world = camera_pos_world
        x_range = np.linspace(-1, 1, scale_size)
        y_range = np.linspace(-1, 1, scale_size)
        x_mat, y_mat = np.meshgrid(x_range, y_range)
        pos = np.stack([x_mat, -y_mat, np.zeros(x_mat.shape)], axis=-1)

        view_dir_world = numpy_norm(camera_pos_world - pos)
        light_dir_world = numpy_norm(light_pos_world - pos)

        light_dir_world = light_dir_world.astype(np.float32)
        view_dir_world = view_dir_world.astype(np.float32)
        return view_dir_world, light_dir_world

    def parse_view_light(name):
        with open(name, 'r') as f:
            lines = f.readlines()
            wlvs = []
            for line in lines[1:]:
                line = line[:-1]

                l, v = line.split()
                camera_pos = [float(i) for i in v.split(',')]
                light_pos = [float(i) for i in l.split(',')]

                item = (light_pos, camera_pos)
                wlvs.append(item)
            return wlvs

    def record(camera_pos):
        Wlvs.append(generate(camera_pos))
        camera_light_pos.append([camera_pos, camera_pos])

    lv_pos = parse_view_light(filename)

    Wlvs = []
    camera_light_pos = []
    for lv in lv_pos:
        light_pos, camera_pos = lv
        record(camera_pos)
    return Wlvs, camera_light_pos




