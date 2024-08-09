# Visualizing and Understanding Atari Agents | Sam Greydanus | 2017 | MIT License

from __future__ import print_function
import warnings;

warnings.filterwarnings('ignore')  # mute warnings, live dangerously ;)

import torch
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import resize

prepro = lambda img: resize(img, (42, 42)).astype(np.float32)
searchlight = lambda I, mask: I * mask + gaussian_filter(I, sigma=3) * (1 - mask)  # choose an area NOT to blur

# occlude 是一个函数，输入是I和mask，输出是I的一个模糊版本，occlude的形状为84*84
occlude = lambda I, mask: I * (1 - mask) + gaussian_filter(I, sigma=3) * mask  # choose an area to blur
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_mask(center, size, r):
    # np.ogrid的作用是生成一个多维的grid,这样就可以生成一个以center为中心，大小为size的网格
    y, x = np.ogrid[-center[0]:size[0] - center[0], -center[1]:size[1] - center[1]]
    # keep是一个布尔矩阵，表示哪些像素点在圆内
    keep = x * x + y * y <= 1
    # 初始化mask矩阵为全0，将圆内的像素点设为1
    mask = np.zeros(size);
    mask[keep] = 1  # select a circle of pixels
    # 将mask模糊化，这里的模糊化是为了让mask的边缘更加平滑
    mask = gaussian_filter(mask, sigma=r)  # blur the circle of pixels. this is a 2D Gaussian for r=r^2=1
    # 将mask归一化到0-1之间
    mask /= mask.max()
    mask = resize(mask, output_shape=[84, 84], mode='constant', anti_aliasing=True).astype(np.float32)
    return mask


def score_frame(model, input, obs, r, d, interp_func, fudge_factor=100):
    # r: radius of blur
    # d: density of scores (if d==1, then get a score for every pixel...
    #    if d==2 then every other, which is 25% of total pixels for a 2D image)
    # 获取原始输出 1 4
    L = model(input)
    input_shape = input.shape
    obs_shape = obs.shape
    len = int(input_shape[-1] / 2)
    # 初始化分数矩阵 17 17
    scores = np.zeros((input_shape[0], int(len / d) + 1, int(len / d) + 1))  # saliency scores S(t,i,j)
    # 横着遍历每个间隔为d的像素
    for k in range(input_shape[0]):
        for i in range(0, len, d):
            # 竖着遍历每个间隔为d的像素
            for j in range(0, len, d):
                # 以i,j为中心获取遮罩，大小为84*84，模糊半径为r,mask数据类型为 ndarray
                mask = get_mask(center=[i, j], size=[len, len], r=r)
                # 以遮罩模糊输入图像 input 1 4 84 84 ,input.squeeze()[-1]大小为84*84，需要把输入转化为ndarray
                # im = interp_func(prepro(input.squeeze()[-1].cpu().detach().numpy()), mask).reshape(1, 84, 84)
                im = interp_func(input[k][-1].cpu().detach().numpy(), mask).reshape(1, 84, 84)
                im = np.repeat(im, 4, axis=0)
                im = torch.from_numpy(im).float().unsqueeze(0).to(device)
                # 获取模糊后的输出
                l = model(im)
                # 计算两个输出之间的差异 .mul_(.5)的意思是乘以0.5
                scores[k, int(i / d), int(j / d)] = (L - l).pow(2).sum().mul_(.5).item()
    pmax = scores.max()
    # 将分数矩阵resize到84*84
    scores = resize(scores, output_shape=[input_shape[0], obs_shape[1], obs_shape[2]], mode='constant',
                    anti_aliasing=True).astype(np.float32)
    # resize函数可能会改变scores数组的值的范围 这里将分数矩阵归一化到0到pmax之间
    # scores = resize(scores, output_shape=[84, 84], mode='reflect', anti_aliasing=True).astype(np.float32)
    scores -= scores.min()
    scores = fudge_factor * pmax * scores / scores.max()
    scores = torch.from_numpy(scores).float().to(device)
    sa_max = scores.max()
    sa_min = scores.min()
    scores = ((scores - sa_min) * 2 / (sa_max - sa_min)) - 1
    scores = scores.unsqueeze(-1).repeat(1, 1, 1, 3)
    return scores


def saliency_on_atari_frame(saliency, atari, fudge_factor, channel=-1, sigma=0):
    # sometimes saliency maps are a bit clearer if you blur them
    # slightly...sigma adjusts the radius of that blur
    # 获取saliency的最大值
    pmax = saliency.max()
    # 将saliency 用插值resize到84*84
    S = resize(saliency, output_shape=[84, 84], mode='reflect', anti_aliasing=True).astype(np.float32)
    # 如果sigma不为0，则给S加高斯模糊，否则不加
    S = S if sigma == 0 else gaussian_filter(S, sigma=sigma)
    # 这是一种数据预处理的方式，通常被称为“零均值化”或“中心化”，它可以将数据的中心移动到原点。
    # 这样做的目的是为了消除数据的偏移，使得数据在数值上更加稳定，便于后续的计算和比较。
    S -= S.min()
    # 将S归一化到0到fudge_factor* pmax之间
    S = fudge_factor * pmax * S / S.max()
    # I为输入图像input,将S加到I的channel通道上,astype('uint16')是为了防止溢出,
    # I为 1 4 84 84的tensor，S为84*84的ndarray，如何把S加到I的channel通道上呢？
    I = atari.squeeze().cpu().detach().numpy()
    I[channel, :, :] += S
    # 将I的值限制在1到255之间
    I /= 255.0
    return torch.from_numpy(I).float().unsqueeze(0).to(device).unsqueeze(0)


def goh(eval_model, input, img_file):
    radius = 2
    density = 2
    with torch.no_grad():
        # Get device of input (i.e., GPU).
        # 如果input是tuple 将其转化为tensor
        if isinstance(input, tuple):
            input = torch.cat(input, dim=0)
        if isinstance(img_file, tuple):
            img_file = torch.cat(img_file, dim=0)
        saliency = score_frame(eval_model, input,img_file,  radius, density, interp_func=occlude)
    # attribution = saliency_on_atari_frame(saliency, input, fudge_factor=100, channel=0)
    return saliency
