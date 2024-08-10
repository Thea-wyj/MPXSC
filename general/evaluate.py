import numpy as np
import torch
import math

from captum.metrics import infidelity
from torch.nn import CosineSimilarity

from GohTest.saliency import goh
from MFPP.test import mfpp
from MFPP.test_mpx import mpx
from sarfa.sarfa_saliency import sarfa

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torch.nn.functional as F

# 把一个>0且 越大越好的值 变化为 取值为0<x<1 的越小越好的值
def transform_value(value):
    return 1 / (1 + value)

def mse_diff(tensor1, tensor2):
    diff_sq = torch.pow(tensor1 - tensor2, 2)
    squared_diff = torch.sum(diff_sq)
    return squared_diff


# 绝对值差计算
def abs_diff(tensor1, tensor2):
    diff = torch.abs(tensor1 - tensor2)
    absolute_diff = torch.sum(diff)
    return absolute_diff


def obs2State(obs):
    # 将原图和掩码相乘，求其输出结果 inp.unsqueeze(0)
    resized_tensor = F.interpolate(obs[:, :, :, 0:1].unsqueeze(1).squeeze(-1), size=(84, 84), mode='bilinear',
                                   align_corners=False)
    # Step 3: Replicate the tensor
    # Expand the tensor to have 4 identical channels
    output_tensor = resized_tensor.expand(-1, 4, -1, -1).clone()
    return output_tensor;


# 获取sufficiency指标值
def get_sufficiency(attribution, threshold, model, input, pred, score):
    # attribution 1,210 160 3
    mask = attribution.clamp(min=threshold)

    mask[mask <= threshold] = 0
    mask[mask > threshold] = 1
    mask = mask.to(torch.float32)
    # input 1 4 84 84
    mask_input = obs2State(mask) * input

    mask_out = model(mask_input)
    ccd = abs_diff(mask_out[0][pred.data.item()], score[0])
    return transform_value(ccd)


def perturb_fn(input):
    noise = torch.tensor(np.random.normal(0, 0.003, input.shape)).float().to(DEVICE)
    return noise, input - noise


# 获取infidelity指标值
def get_infidelity(model, input, attribution, pred):
    # Computes infidelity score for saliency maps
    infid = infidelity(model, perturb_fn, input, obs2State(attribution), target=pred.data.item())
    return torch.sigmoid(infid)


# 获取sensitivity指标值
def get_sensitivity(attribution, anti_attribution):
    cs = CosineSimilarity()  # 取值为-1,1,越大相似度越高，越小相似度越低
    sim = cs.forward(x1=attribution.view(attribution.shape[0], -1),
                     x2=anti_attribution.view(anti_attribution.shape[0], -1))
    # 取值为（0~1）越大相似度越高，越小相似度越低
    return (sim + 1) / 2


# 获取stability指标值
def get_stability(sensitivity_process):
    sens = sensitivity_process
    return transform_value(sens)


# 获取validity指标值
def get_validity(attribution, threshold, model, input, pred, score):
    # attribution 1,4,84,84
    mask = attribution.clamp(max=threshold)

    mask[mask >= threshold] = 0
    mask[mask < threshold] = 1
    mask = mask.to(torch.float32)
    # input 1 4 84 84
    mask_input = obs2State(mask) * input

    mask_out = model(mask_input)
    val_ig = abs_diff(mask_out[0][pred.data.item()], score[0])
    return torch.sigmoid(val_ig)


# def explain_fn(input,explain, **kwargs):
#     return explain.attribute(input, **kwargs)


def mfpp_explain_fn(image, model=None, img_file=None, target=0):
    return (mfpp(model, image, img_file, target),)


def mpx_explain_fn(image, model=None, img_file=None, target=0):
    return (mpx(model, img_file, image, target),)


def goh_explain_fn(image, model=None, img_file=None):
    return (goh(model, image, img_file),)


def sarfa_explain_fn(image, model=None, target=0, img_file=None):
    return (sarfa(model, image, img_file, target),)
