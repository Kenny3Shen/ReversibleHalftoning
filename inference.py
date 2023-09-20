import argparse
import os
from collections import OrderedDict
from glob import glob
from os.path import join

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model.model import Quantize
from model.model import ResHalf
from utils import util


class Inferencer:
    def __init__(self, checkpoint_path, model, use_cuda=False, multi_gpu=False):
        self.checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.use_cuda = use_cuda
        self.model = model.eval()
        if multi_gpu:
            self.model = torch.nn.DataParallel(self.model)
            state_dict = self.checkpoint['state_dict']
        else:
            # remove keyword "module" in the state_dict
            state_dict = OrderedDict()
            for k, v in self.checkpoint['state_dict'].items():
                name = k[7:]
                state_dict[name] = v
        if self.use_cuda:
            self.model = self.model.cuda()
        self.model.load_state_dict(state_dict)

    def __call__(self, input_img, decoding_only=False):
        # 禁用梯度计算
        with torch.no_grad():
            scale = 8  # 定义缩放因子
            _, _, H, W = input_img.shape  # 获取输入图像的高度和宽度
            # 如果高度或宽度不能被缩放因子整除，就进行反射填充
            if H % scale != 0 or W % scale != 0:
                input_img = F.pad(input_img, [0, scale - W % scale, 0, scale - H % scale], mode='reflect')
            if self.use_cuda:
                input_img = input_img.cuda()
            if decoding_only:
                # 调用self.model进行图像解码
                # model=ResHalf(train=False)
                # print(len(input_img))
                resColor = self.model(input_img, decoding_only)
                # print(type(resColor))
                # 如果之前进行了填充，将结果裁剪回原始图像大小
                if H % scale != 0 or W % scale != 0:
                    resColor = resColor[:, :, :H, :W]
                # 返回逆半调图像
                return resColor
            else:
                # 调用self.model进行图像处理，获取两个结果：resHalftone和resColor
                resHalftone, resColor = self.model(input_img, decoding_only)
                # 对resHalftone进行量化操作
                # Q:为什么在 ResHalf 量化过了这里又进行一次量化
                resHalftone = Quantize.apply((resHalftone + 1.0) * 0.5) * 2.0 - 1.
                # 如果之前进行了填充，将结果裁剪回原始图像大小
                if H % scale != 0 or W % scale != 0:
                    resHalftone = resHalftone[:, :, :H, :W]
                    resColor = resColor[:, :, :H, :W]
                # 返回半调图像和逆半调图像
                return resHalftone, resColor


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='invHalf')
    parser.add_argument('--model', default="checkpoints/model_best.pth.tar", type=str,
                        help='model weight file path')
    parser.add_argument('--decoding', action='store_true', default=False, help='restoration from halftone input')
    parser.add_argument('--data_dir', default="./test_imgs", type=str,
                        help='where to load input data (RGB images)')
    parser.add_argument('--save_dir', default="./result", type=str,
                        help='where to save the result')
    args = parser.parse_args()

    data_dir = os.path.join(args.data_dir)
    save_dir = os.path.join(args.save_dir)
    invhalfer = Inferencer(
        checkpoint_path=args.model,
        model=ResHalf(train=False)
    )
    util.ensure_dir(save_dir)
    test_imgs = glob(join(args.data_dir, '*.*g'))
    # print(test_imgs)
    print('------loaded %d images.' % len(test_imgs))
    for img in test_imgs:
        print('[*] processing %s ...' % img)
        (name, suffix) = img.split('\\')[-1].split('.')
        if args.decoding:
            # output restored image only
            # Given groups=1, weight of size [64, 4, 3, 3], expected input[1, 1, H, W] to have 4 channels, but got 1 channels instead
            # 出现上述错误是因为没有正确进入 self.model 的 decoder 分支
            # print(np.array(Image.open(img)).shape)
            # 读入灰度图片
            input_img = cv2.imread(img, flags=cv2.IMREAD_GRAYSCALE) / 127.5 - 1.
            c = invhalfer(util.img2tensor(input_img), decoding_only=True)  # __call__
            c = util.tensor2img(c / 2. + 0.5) * 255.
            cv2.imwrite(join(save_dir, 'restore_half_' + name + '.png'), c)
        else:
            # RuntimeError: Given groups=1, weight of size [64, 4, 3, 3], expected input[1, 3, H, W] to have 4 channels, but got 3 channels instead
            # 需要将[H,W,C]中的 C(channel) 修改为4通道 RGBA
            if suffix != 'png':
                Image.open(img).save(f"./{data_dir}/{name}.png")
                os.remove(img)
                img = f"./{data_dir}/{name}.png"
            if np.array(Image.open(img)).shape[-1] != 4:
                Image.open(img).convert("RGBA").save(f"{img}")

            # 读入完整图片包括 alpha 通道
            # 对比源代码此处将 flags 参数从 cv2.IMREAD_COLOR 修改为 cv2.IMREAD_UNCHANGED
            input_img = cv2.imread(img, flags=cv2.IMREAD_UNCHANGED) / 127.5 - 1.
            # print(input_img.shape)
            # img2tensor: 将 NumPy 矩阵[H,W,C](灰度图像没有维度 C) 转换为 PyTorch 4维矩阵 [B,C,H,W]
            h, c = invhalfer(util.img2tensor(input_img), decoding_only=False)
            # tensor2img： PyTorch -> NumPy
            h = util.tensor2img(h / 2. + 0.5) * 255.
            c = util.tensor2img(c / 2. + 0.5) * 255.
            cv2.imwrite(join(save_dir, 'halftone_' + name + '.png'), h)
            cv2.imwrite(join(save_dir, 'restored_' + name + '.png'), c)
            # cv2.imwrite(join(save_dir, 'halftone_' + img.split('/')[-1].split('.')[0] + '.png'), h)
            # cv2.imwrite(join(save_dir, 'restored_' + img.split('/')[-1].split('.')[0] + '.png'), c)
