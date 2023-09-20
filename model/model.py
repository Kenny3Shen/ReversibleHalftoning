import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from .hourglass import HourGlass
from utils.dct import DCT_Lowfrequency
from utils.filters_tensor import bgr2gray

from collections import OrderedDict
import numpy as np


class Quantize(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y = x.round()
        return y

    @staticmethod
    def backward(ctx, grad_output):
        inputX = ctx.saved_tensors
        return grad_output


class ResHalf(nn.Module):
    def __init__(self, train=True, warm_stage=False):
        super(ResHalf, self).__init__()
        # 接收 RGBA 图像(4通道)，输出灰度图像(1通道)
        self.encoder = HourGlass(inChannel=4, outChannel=1, resNum=4, convNum=4)
        # 接收灰度图像，输出 RGB 图像
        self.decoder = HourGlass(inChannel=1, outChannel=3, resNum=4, convNum=4)
        # 执行离散余弦变换（DCT）
        if train:
            self.dcter = DCT_Lowfrequency(size=256, fLimit=50)
        # 量化数据 quantize [-1,1] data to be {-1,1}
        self.quantizer = lambda x: Quantize.apply(0.5 * (x + 1.)) * 2. - 1.
        self.isTrain = train
        if warm_stage:
            for name, param in self.decoder.named_parameters():
                # 参数在训练过程中将不会更新
                param.requires_grad = False

    # 向输入添加脉冲噪声 params = (半色调图像, 概率p)
    def add_impluse_noise(self, input_halfs, p=0.0):
        # N：Batch，批处理大小，表示一个batch中的图像数量
        # C：Channel，通道数，表示一张图像中的通道数
        # H：Height，高度，表示图像垂直维度的像素数
        # W：Width，宽度，表示图像水平维度的像素数
        N, C, H, W = input_halfs.shape
        SNR = 1 - p
        np_input_halfs = input_halfs.detach().to("cpu").numpy()
        np_input_halfs = np.transpose(np_input_halfs, (0, 2, 3, 1))
        for i in range(N):
            mask = np.random.choice((0, 1, 2), size=(H, W, 1), p=[SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
            np_input_halfs[i, mask == 1] = 1.0
            np_input_halfs[i, mask == 2] = -1.0
        return torch.from_numpy(np_input_halfs.transpose((0, 3, 1, 2))).to(input_halfs.device)

    # 前向传播
    def forward(self, *x):
        # 原代码中并没有正确接收 decoding_only 参数
        # x[0]: ref_halftone 半调图像
        # x[1]: decoding_only
        # print(x[0].shape, x[1]) -> (tensor, bool)
        # noise = torch.randn_like(x[1]) * 0.3
        if not x[1]:
            # halfRes = self.encoder(torch.cat((x[0], noise), dim=1))
            halfRes = self.encoder(x[0])
            # halfRes = self.encoder(torch.cat((input_tensor+noise_map, input_tensor-noise_map), dim=1))
            halfResQ = self.quantizer(halfRes)
            # ! for testing only
            # halfResQ = self.add_impluse_noise(halfResQ, p=0.20)
            restored = self.decoder(halfResQ)
        else:
            restored = self.decoder(x[0])
            return restored

        if self.isTrain:  # Train
            halfDCT = self.dcter(halfRes / 2. + 0.5)
            refDCT = self.dcter(bgr2gray(x[0] / 2. + 0.5))
            return halfRes, halfDCT, refDCT, restored
        else:
            return halfRes, restored
