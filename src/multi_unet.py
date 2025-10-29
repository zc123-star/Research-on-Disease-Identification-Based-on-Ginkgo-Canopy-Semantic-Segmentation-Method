import torch
import torch.nn as nn
import torch.nn.functional as F

class HaarWaveletDownsampling(nn.Module):
    def __init__(self, in_channels):
        super(HaarWaveletDownsampling, self).__init__()
        self.in_channels = in_channels

        # 定义 Haar Wavelet 低通滤波器（平均）
        self.register_buffer('low_pass_filter', torch.tensor([[1/2., 1/2.], [1/2., 1/2.]]))
        # 定义三个方向的高通滤波器
        self.register_buffer('high_pass_filter_h', torch.tensor([[-1/2., -1/2.], [1/2., 1/2.]]))  # 水平
        self.register_buffer('high_pass_filter_v', torch.tensor([[-1/2., 1/2.], [-1/2., 1/2.]]))  # 垂直
        self.register_buffer('high_pass_filter_d', torch.tensor([[1/2., -1/2.], [-1/2., 1/2.]]))  # 对角

        # 重复滤波器以匹配输入通道数
        self.low_pass_filter = self.low_pass_filter.view(1, 1, 2, 2).repeat(self.in_channels, 1, 1, 1)
        self.high_pass_filter_h = self.high_pass_filter_h.view(1, 1, 2, 2).repeat(self.in_channels, 1, 1, 1)
        self.high_pass_filter_v = self.high_pass_filter_v.view(1, 1, 2, 2).repeat(self.in_channels, 1, 1, 1)
        self.high_pass_filter_d = self.high_pass_filter_d.view(1, 1, 2, 2).repeat(self.in_channels, 1, 1, 1)

        # 定义1x1卷积、批归一化和ReLU序列
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_channels*4, in_channels, kernel_size=1, stride=1),  # 注意输入通道数是 in_channels * 4
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        device = x.device
        # 将滤波器移动到输入数据的设备上
        low_pass_filter = self.low_pass_filter.to(device)
        high_pass_filter_h = self.high_pass_filter_h.to(device)
        high_pass_filter_v = self.high_pass_filter_v.to(device)
        high_pass_filter_d = self.high_pass_filter_d.to(device)

        # 使用低通滤波器获取低频分量
        low_freq = F.conv2d(x, low_pass_filter, stride=2, groups=self.in_channels)
        # 使用高通滤波器获取三个方向的高频分量
        high_freq_h = F.conv2d(x, high_pass_filter_h, stride=2, groups=self.in_channels)
        high_freq_v = F.conv2d(x, high_pass_filter_v, stride=2, groups=self.in_channels)
        high_freq_d = F.conv2d(x, high_pass_filter_d, stride=2, groups=self.in_channels)

        # 拼接低频和三个方向的高频分量
        concatenated = torch.cat([low_freq, high_freq_h, high_freq_v, high_freq_d], dim=1)

        # 通过1x1卷积降维，使通道数回到原来的大小
        x = self.conv_bn_relu(concatenated)

        return x
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y

class SEDoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(SEDoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            SEBlock(out_channels)

        )

class MultiGridConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=[1, 2, 4]):
        super(MultiGridConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation_rates[0], dilation=dilation_rates[0])
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation_rates[1], dilation=dilation_rates[1])
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation_rates[2], dilation=dilation_rates[2])
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = x1 + x2 + x3
        x = self.bn(x)
        return x


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            HaarWaveletDownsampling(in_channels),
            MultiGridConv(in_channels, out_channels),
            SEDoubleConv(out_channels, out_channels)

            #nn.MaxPool2d(2, stride=2)
            #DepthwiseSeparableConv(out_channels, out_channels),
        )

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = SEDoubleConv(in_channels, out_channels, in_channels // 2)

        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = SEDoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

class Multi_SE_UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, bilinear=True, base_c=64):
        super(Multi_SE_UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = SEDoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)

        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)


    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)


        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)
        return {"out": logits}