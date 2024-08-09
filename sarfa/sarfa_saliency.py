from __future__ import print_function
import warnings

import math
import numpy as np
from scipy.stats import entropy, wasserstein_distance
from scipy.spatial.distance import jensenshannon

# Visualizing and Understanding Atari Agents | Sam Greydanus | 2017 | MIT License

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


def score_frame(model, input, obs, target, r, d, interp_func, fudge_factor=100):
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
                scores[k, int(i / d), int(j / d)] = computeSaliencyUsingSarfa(target, L[k].unsqueeze(0), l)
    pmax = scores.max()
    # 将分数矩阵resize到84*84
    scores = resize(scores, output_shape=[input_shape[0], obs_shape[1],  obs_shape[2]], mode='constant',
                    anti_aliasing=True).astype(np.float32)
    # resize函数可能会改变scores数组的值的范围 这里将分数矩阵归一化到0到pmax之间
    # scores = resize(scores, output_shape=[84, 84], mode='reflect', anti_aliasing=True).astype(np.float32)
    scores -= scores.min()
    scores = fudge_factor * pmax * scores / scores.max()
    scores = torch.from_numpy(scores).float().to(device)
    sa_max = scores.max()
    sa_min = scores.min()
    scores = ((scores - sa_min) * 2 / (sa_max - sa_min)) - 1
    scores = scores.unsqueeze(-1).repeat(1,1,1,3)
    return scores


def your_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def cross_entropy(dictP, dictQ, original_action):
    """
    This function calculates normalized cross entropy (KL divergence) of Q-values of state Q wrt state P.
    Input:
        dictP: Q-value dictionary of perturbed state
        dictQ: Q-value dictionary of original state
    Output:p = policy[:best_move+1]
    p = np.append(p, policy[best_move+1:])

        K: normalized cross entropy
    """
    Dpq = 0.
    Q_p = []  # values of moves in dictP^dictQ wrt P
    Q_q = []  # values of moves in dictP^dictQ wrt Q
    # 创建一个包含所有你想要的索引的列表
    indices = [i for i in range(dictP.shape[1]) if i != original_action]
    Q_p = dictP[:, indices].squeeze().tolist()
    Q_q = dictQ[:, indices].squeeze().tolist()
    # for move in dictP:
    #     if move == original_action:
    #         print('skipping original action for KL-Divergence')
    #         continue
    #     if move in dictQ:
    #         Q_p.append(dictP[move])
    #         Q_q.append(dictQ[move])
    # # converting Q-values into probability distribution
    Q_p = your_softmax(np.asarray(Q_p))
    Q_q = your_softmax(np.asarray(Q_q))
    KL = entropy(Q_q, Q_p)
    # KL = wasserstein_distance(Q_q, Q_p)
    # return (KL)/(KL + 1.)
    return 1. / (KL + 1.)


def computeSaliencyUsingSarfa(original_action, q_vals_before_perturbation, q_vals_after_perturbation):
    answer = 0

    # probability of original move in perturbed state
    # print(q_vals_after_perturbation)
    # 扰动状态
    q_value_action_perturbed_state = q_vals_after_perturbation[:, original_action].data.item()
    # 原始状态
    q_value_action_original_state = q_vals_before_perturbation[:, original_action].data.item()
    # 扰动Q值
    q_values_after_perturbation = q_vals_after_perturbation.squeeze().tolist()
    # 原始q值
    q_values_before_perturbation = q_vals_before_perturbation.squeeze().tolist()
    # 扰动状态的概率
    probability_action_perturbed_state = np.exp(q_value_action_perturbed_state) / np.sum(
        np.exp(q_values_after_perturbation))
    # 原始状态的概率
    probability_action_original_state = np.exp(q_value_action_original_state) / np.sum(
        np.exp(q_values_before_perturbation))

    K = cross_entropy(q_vals_after_perturbation, q_vals_before_perturbation, original_action)

    dP = probability_action_original_state - probability_action_perturbed_state

    # if probability_action_perturbed_state < probability_action_original_state:  # harmonic mean
    answer = 2 * dP * K / (dP + K)

    # QmaxAnswer = computeSaliencyUsingQMaxChange(original_action, q_vals_before_perturbation,
    #                                             q_vals_after_perturbation)
    # action_gap_before_perturbation, action_gap_after_perturbation = computeSaliencyUsingActionGap(
    #     q_vals_before_perturbation, q_vals_after_perturbation)

    # print("Delta P = ", dP)
    # print("KL normalized = ", K)
    # print("KL normalized inverse = ", 1/K)
    # print(entry['saliency'])
    # return answer, dP, K, QmaxAnswer, action_gap_before_perturbation, action_gap_after_perturbation
    return answer


def computeSaliencyUsingQMaxChange(original_action, q_vals_before_perturbation, q_vals_after_perturbation):
    answer = 0

    # best_action = None
    best_q_value, best_action = torch.max(q_vals_after_perturbation.data, 1)

    # for move, q_value in q_vals_after_perturbation.items():
    #     if best_action is None:
    #         best_action = move
    #         best_q_value = q_value
    #     elif q_value > best_q_value:
    #         best_q_value = q_value
    #         best_action = move

    if best_action != original_action:
        answer = 1

    return answer


def computeSaliencyUsingActionGap(q_vals_before_perturbation, q_vals_after_perturbation):
    q_vals_before_perturbation = sorted(q_vals_before_perturbation.squeeze().tolist())
    q_vals_after_perturbation = sorted(q_vals_after_perturbation.squeeze().tolist())
    action_gap_before_perturbation = q_vals_before_perturbation[-1] - q_vals_before_perturbation[-2]
    action_gap_after_perturbation = q_vals_after_perturbation[-1] - q_vals_after_perturbation[-2]

    return action_gap_before_perturbation, action_gap_after_perturbation


def sarfa(model, input, img_file, target):
    radius = 2
    density = 2
    with torch.no_grad():
        # Get device of input (i.e., GPU).
        # 如果input是tuple 将其转化为tensor
        if isinstance(input, tuple):
            input = torch.cat(input, dim=0)
        if isinstance(img_file, tuple):
            img_file = torch.cat(img_file, dim=0)
        saliency = score_frame(model, input, img_file, target, radius, density, interp_func=occlude)
    # attribution = saliency_on_atari_frame(saliency, input, fudge_factor=100, channel=0)
    return saliency
