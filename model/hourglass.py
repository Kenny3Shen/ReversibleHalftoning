import torch.nn as nn
from .base_module import ConvBlock, DownsampleBlock, ResidualBlock, SkipConnection, UpsampleBlock


class HourGlass(nn.Module):
    # 卷积层conv 残差网络层 resNet
    """
    论文 3.1 Network Architecture
    抖动网络和恢复网络都采用 U 形架构。两个网络具有相似的结构，
    包含三个下采样块、三个上采样块、四个残差块和两个卷积块。
    采用 U-Net 作为网络主干只是因为它扩大了感受野，其他合格的 CNN 架构也可能有效。 Q: 是否有性能更好的 CNN 架构？
    我们提出了抖动网络的两个关键设计，即噪声激励块(NIB)和二元门(binary gate)，这使得 CNN 能够正确地模拟半色调。
    """

    def __init__(self, convNum=4, resNum=4, inChannel=6, outChannel=3):
        super(HourGlass, self).__init__()
        self.inConv = ConvBlock(inChannel, 64, convNum=2)
        self.down1 = nn.Sequential(*[DownsampleBlock(64, 128, withConvRelu=False), ConvBlock(128, 128, convNum=2)])
        self.down2 = nn.Sequential(
            *[DownsampleBlock(128, 256, withConvRelu=False), ConvBlock(256, 256, convNum=convNum)])
        self.down3 = nn.Sequential(
            *[DownsampleBlock(256, 512, withConvRelu=False), ConvBlock(512, 512, convNum=convNum)])

        self.residual = nn.Sequential(*[ResidualBlock(512) for _ in range(resNum)])

        self.up3 = nn.Sequential(*[UpsampleBlock(512, 256), ConvBlock(256, 256, convNum=convNum)])
        self.skip3 = SkipConnection(256)
        self.up2 = nn.Sequential(*[UpsampleBlock(256, 128), ConvBlock(128, 128, convNum=2)])
        self.skip2 = SkipConnection(128)
        self.up1 = nn.Sequential(*[UpsampleBlock(128, 64), ConvBlock(64, 64, convNum=2)])
        self.skip1 = SkipConnection(64)
        self.outConv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, outChannel, kernel_size=1, padding=0)
        )

    """
    1. 输入卷积层 (`inConv`)
       - 作用：接收原始输入图像或特征图，并执行初始卷积操作。
       - 具体功能：将输入数据进行卷积操作，以提取一些基本的特征，例如边缘、颜色等。
    
    2. 下采样层 (`down1`, `down2`, `down3`)
       - 作用：降低特征图的分辨率，从而扩大感受野（感知区域），捕捉更广泛的上下文信息。
       - 具体功能：通过卷积和池化操作，减小特征图的尺寸，同时增加特征图的深度，以获取更抽象的特征表示。
    
    *3. 残差块 (`residual`)
       - 作用：引入残差连接，允许网络跳过一些层级，有助于解决梯度消失问题，提供更稳定的训练。
       - 具体功能：每个残差块包含一系列卷积层，它们以恒等映射（identity mapping）的方式处理特征图，将输入的特征与经过卷积操作的特征相加，以获得更强的特征表示。
    
    4. 上采样层 (`up3`, `up2`, `up1`)
       - 作用：增加特征图的分辨率，将低分辨率的特征图映射回原始输入分辨率，以进行更精细的预测。
       - 具体功能：通过上采样（例如反卷积或插值）操作，将特征图的大小放大，使其与原始输入尺寸匹配。
    
    *5. 跳跃连接 (`skip3`, `skip2`, `skip1`)
       - 作用：将不同层级的特征图相互连接，以保留低层级和高层级特征的信息，有助于精确的位置信息和细节。
       - 具体功能：将上采样的特征图与相应层级的低分辨率特征图进行连接，通过融合不同分辨率的信息，使网络能够更好地理解输入图像的结构和内容。
    
    6. 输出卷积层 (`outConv`)
       - 作用：生成最终的模型输出，通常用于分类、分割或其他任务。
       - 具体功能：对上采样和跳跃连接后的特征图进行卷积操作，以产生模型的最终预测结果。这通常包括对类别的预测、物体边界的分割或其他任务相关的输出。
    
    HourGlass 模型的主要特点是多层级特征提取和跳跃连接，这有助于有效地处理图像任务，同时保留了丰富的细节和上下文信息
    """

    def forward(self, x):
        # 输入图像 x
        f1 = self.inConv(x)  # f1 = 卷积x
        f2 = self.down1(f1)  # f2 = 下采样f1
        f3 = self.down2(f2)  # f3 = 下采样f2
        f4 = self.down3(f3)  # f4 = 下采样f3
        r4 = self.residual(f4)  # r4 = 残差层处理f4 (模型对输入的高级表示)
        r3 = self.skip3(self.up3(r4), f3)  # r3 = 跳跃连接(上采样r4, f3)
        r2 = self.skip2(self.up2(r3), f2)  # r2 = 跳跃连接(上采样r3, f2)
        r1 = self.skip1(self.up1(r2), f1)  # r1 = 跳跃连接(上采样r2, f1)
        y = self.outConv(r1)  # 预测输出图像 y = 输出卷积r1
        return y


class ResidualHourGlass(nn.Module):
    def __init__(self, resNum=4, inChannel=6, outChannel=3):
        super(ResidualHourGlass, self).__init__()
        self.inConv = nn.Conv2d(inChannel, 64, kernel_size=3, padding=1)
        self.residualBefore = nn.Sequential(*[ResidualBlock(64) for _ in range(2)])
        self.down1 = nn.Sequential(
            *[DownsampleBlock(64, 128, withConvRelu=False), ConvBlock(128, 128, convNum=2)])
        self.down2 = nn.Sequential(
            *[DownsampleBlock(128, 256, withConvRelu=False), ConvBlock(256, 256, convNum=2)])
        self.residual = nn.Sequential(*[ResidualBlock(256) for _ in range(resNum)])
        self.up2 = nn.Sequential(*[UpsampleBlock(256, 128), ConvBlock(128, 128, convNum=2)])
        self.skip2 = SkipConnection(128)
        self.up1 = nn.Sequential(*[UpsampleBlock(128, 64), ConvBlock(64, 64, convNum=2)])
        self.skip1 = SkipConnection(64)
        self.residualAfter = nn.Sequential(*[ResidualBlock(64) for _ in range(2)])
        self.outConv = nn.Sequential(
            nn.Conv2d(64, outChannel, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        f1 = self.inConv(x)
        f1 = self.residualBefore(f1)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        r3 = self.residual(f3)
        r2 = self.skip2(self.up2(r3), f2)
        r1 = self.skip1(self.up1(r2), f1)
        y = self.residualAfter(r1)
        y = self.outConv(y)
        return y
