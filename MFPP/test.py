import json
import os

import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from matplotlib import pyplot as plt
from skimage.segmentation import slic
from tqdm import trange


def mfpp(model,
         input,
         img_file,
         target=None,
         seed=0,
         num_masks=20000,
         resize_offset=2.2,
         layer=5,
         batch_size=32,
         p_1=0.5,
         resize_mode='bilinear'):
    r"""MFPP.

    Args:
        model (:class:`torch.nn.Module`): a model.
        input (:class:`torch.Tensor`): input tensor.
        seed (int, optional): manual seed used to generate random numbers.
            Default: ``0``.
        num_masks (int, optional): number of MFPP random masks to use.
            Default: ``8000``.
        resize_offset(float): the offset for resized image for crop. Default: ``2.5``.
        layer (int): the number of segments style. Default: ``5``.
        batch_size (int, optional): batch size to use. Default: ``128``.
        p_1 (float, optional): with prob p_1, a low-res cell is set to 1;
            otherwise, it's 1. Default: ``0.5``.
        resize_mode (str, optional): If resize is not None, use this mode for
            the resize function. Default: ``'bilinear'``.

    Returns:
        :class:`torch.Tensor`: MFPP saliency map.
    """
    SEG_COEFF = 100  # relationship between segmente number and index. It's fixed to 50.

    # img_file ndarray 4 84 84
    img = img_file.transpose(1, 2, 0)
    img = img[:, :, -1:]
    # img ndarray 84 84 1

    IMAGE_SIZE = (84, 84, 1)

    segments = [[], []] * layer  # layer组超像素
    n_features = [[]] * layer  # layer组超像素的特征数，即超像素的数量

    for i in range(0, layer):
        num_segments = SEG_COEFF * (2 ** i)  # 第i层超像素数量 50*2^i 第一层 50 第二层 100个 第三层 200个 第四层 400个 第五层 800个
        # 初始化每一组超像素
        segments[i] = slic(img, n_segments=num_segments, compactness=10, sigma=4)
        # n_features和n_segments相比，n_features是实际的超像素数量，n_segments是理论的超像素数量
        n_features[i] = np.unique(np.asarray(segments[i])).shape[0]

    with torch.no_grad():
        # Get device of input (i.e., GPU).
        # 如果input是tuple 将其转化为tensor
        if isinstance(input, tuple):
            input = torch.cat(input, dim=0)
        dev = input.device
        # Initialize saliency mask and mask normalization term.
        # input_shape = tensor (1, 4, 84, 84)
        input_shape = input.shape
        # saliency_shape = list [1, 4, 84, 84]
        saliency_shape = list(input_shape)

        height = input_shape[2]  # 84
        width = input_shape[3]  # 84

        H = height + math.floor(resize_offset * height)  # 268
        W = width + math.floor(resize_offset * width)  # 268

        # out tensor 32,4
        out = model(input)
        # 输出类的个数
        num_classes = out.shape[1]

        saliency_shape[1] = num_classes
        # 初始化为全0
        # saliency tensor(1,4,84,84)
        saliency = torch.zeros(saliency_shape, device=dev)

        # Save current random number generator state.
        state = torch.get_rng_state()

        # Set seed.
        torch.manual_seed(seed)  # seed = 0
        np.random.seed(seed)

        # num_chunks代表一共有多少个batch 计算方式为 随机掩码的数量/每个batch的数量
        num_chunks = num_masks // batch_size
        # print("num_chunks:", num_chunks)  # 188
        # layer_group代表每一组有多少个batch //代表向下取整 例如 5//2 = 2 上取整的符号是
        layer_group = math.ceil(num_chunks / layer)
        # print("layer_group:", layer_group)  # 37

        # 生成num_chunks个掩码
        for chunk in trange(num_chunks):
            # mask_bs代表这个轮次里面裁切的数量
            mask_bs = min(num_masks - batch_size * chunk, batch_size)
            # print("mask_bs ",mask_bs) #32
            # layer_index代表当前是第几组
            layer_index = chunk // layer_group
            # print("layer_index:",layer_index) #0~4
            # np_masks代表裁切后的掩码
            np_masks = np.zeros((mask_bs,) + IMAGE_SIZE[:2], dtype=np.float32)
            #             print("init_masks.shape:",np_masks.shape) # (32, 224, 224)
            # 随机指定超像素的值为0或1，并记录随机结果
            data = np.random.choice([0, 1], size=n_features[layer_index], p=[1 - p_1, p_1])
            #                 print("data:",data)
            # 找到data中为0的索引
            zeros = np.where(data == 0)[0]
            # print("zeros:",zeros)
            # 初始化掩码和分割结果大小一致
            mask = np.zeros(segments[layer_index].shape).astype(float)
            # 按照随机值为0的索引将对应超像素块中的像素值设为1
            for z in zeros:
                #               print("z:",z)
                mask[segments[layer_index] == z] = 1.0
            #               print("mask:",mask)
            # 将mask转化为Image对象
            mask = Image.fromarray(mask * 255.)
            #               plt.imshow(mask)

            # 将mask resize到HxW，Image.BILINEAR代表双线性插值
            mask = mask.resize((H, W), Image.BILINEAR)
            mask = np.array(mask)
            for i in range(mask_bs):  # (32 masks)
                # crop to HxW
                # w_crop和h_crop代表随机裁剪的起始位置
                w_crop = np.random.randint(0, resize_offset * width + 1)
                h_crop = np.random.randint(0, resize_offset * height + 1)
                #                 print("w_crop h_crop:",w_crop,h_crop)
                # 进行裁切，裁切大小为原图大小
                np_masks[i] = mask[h_crop:height + h_crop, w_crop:width + w_crop]
                #                 print("mask.shape:",mask.shape)
                #                 print("{}:{} {}:{}",h_crop,height + h_crop, w_crop,width + w_crop)
                # 如果裁切后的掩码中有nan值，将其替换为第一个掩码的值，如果第一个掩码也有nan值，将其替换为第二个掩码的值
                if np.isnan(np.sum(np_masks[i])):
                    np_masks[i] = np_masks[0].copy() if not np.isnan(np.sum(np_masks[0])) else np_masks[1].copy()
                #                 np_masks[i] /= np.max(np_masks[i])
                # 将裁切的掩码转化为0-1之间的值
                np_masks[i] /= 255.0

            # masks tensor (32 1 84 84)
            masks = torch.from_numpy(np_masks)
            masks = masks.to(dev)
            # 将掩码resize到84 84
            masks = masks.resize(batch_size, 1, 84, 84)

            # numerate(input)返回一个迭代器，迭代器中的每个元素都是一个tuple，tuple中第一个元素是索引，第二个元素是input中的元素
            for i, inp in enumerate(input):
                # 将原图和掩码相乘，求其输出结果 inp.unsqueeze(0)
                out = torch.sigmoid(model(inp.unsqueeze(0) * masks))
                if len(out.shape) == 4:
                    assert out.shape[2] == 1
                    assert out.shape[3] == 1
                    out = out[:, :, 0, 0]
                # torch.matmul代表矩阵相乘 out.data.transpose(0, 1)代表将out.data的维度转化为(1,num_classes)，masks.view(mask_bs, height * width)代表将masks的维度转化为(mask_bs, height * width)，将二者相乘可以得到每个掩码
                sal = torch.matmul(out.data.transpose(0, 1),
                                   masks.view(mask_bs, height * width))
                sal = sal.view((num_classes, height, width))

                saliency[i] = saliency[i] + sal

        # 将saliency除以num_masks，得到平均值
        saliency /= num_masks

        # Restore original random number generator state.
        saliency = saliency[:, target]
        sa_max = saliency.max()
        sa_min = saliency.min()
        saliency = ((saliency - sa_min) * 2 / (sa_max - sa_min)) - 1
        saliency = saliency.repeat(4, 1, 1).unsqueeze(0)

    return saliency
