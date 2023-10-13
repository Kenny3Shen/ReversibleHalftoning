import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function

from utils.dct import DCT_Lowfrequency
from utils.filters_tensor import bgr2gray
from .hourglass import HourGlass


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
        # 接收叠加了高斯噪声的 RGB 图像(4通道)，输出灰度图像(1通道)
        self.encoder = HourGlass(inChannel=4, outChannel=1, resNum=4, convNum=4)
        # 接收灰度图像，输出 RGB 图像
        self.decoder = HourGlass(inChannel=1, outChannel=3, resNum=4, convNum=4)
        # 执行离散余弦变换（DCT）
        if train:
            self.dcter = DCT_Lowfrequency(size=256, fLimit=50)
        # 量化数据 quantize [-1,1] data to be {0,1}
        self.quantizer = lambda x: Quantize.apply(0.5 * (x + 1.)) * 2. - 1.
        self.isTrain = train
        if warm_stage:
            for name, param in self.decoder.named_parameters():
                # 参数在热身过程中将不会更新
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
    def forward(self, input_img, decoding_only=False):
        if decoding_only:
            halfResQ = self.quantizer(input_img)
            restored = self.decoder(halfResQ)
            return restored

        noise = torch.randn_like(input_img) * 0.   # 0.3 可能与蓝噪声损失系数有关
        # print("noise shape: ", noise.shape)
        noise_zero = torch.zeros_like(input_img) * 0.3
        noise_one = torch.ones_like(input_img) * 0.3
        halfNoise = torch.cat((input_img, noise[:, :1, :, :]), dim=1)
        cv2.imwrite("halfNoise.png", halfNoise[0].detach().cpu().numpy().transpose((1, 2, 0)) * 255)
        halfRes = self.encoder(halfNoise)
        halfResQ = self.quantizer(halfRes)
        restored = self.decoder(halfResQ)

        if self.isTrain:
            halfDCT = self.dcter(halfRes / 2. + 0.5)
            refDCT = self.dcter(bgr2gray(input_img / 2. + 0.5))
            return halfRes, halfDCT, refDCT, restored
        else:
            return halfRes, restored
