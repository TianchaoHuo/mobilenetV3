import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F



class hswish(nn.Module):
    def __init__(self, inplace=True):
        super(hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3, inplace=self.inplace) / 6


class hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3, inplace=self.inplace) / 6


#----------------------------------------------------Coordinate attention Module ---------------------------------------------------------#

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = hswish()

    def forward(self, x):
        identity = x
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y) 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        y = identity * x_w * x_h

        return y
#----------------------------------------------------SE Module ---------------------------------------------------------#

class SeModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SeModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            hsigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        
        return x * y.expand_as(x)

# class SeModule(nn.Module):
#     def __init__(self, in_size, reduction=4):
#         super(SeModule, self).__init__()
#         self.se = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1), 
#             nn.Conv2d(in_size, in_size//reduction, kernel_size=1, stride=1, padding=0, bias=False),
#             #nn.BatchNorm2d(in_size // reduction),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_size//reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
#             #nn.BatchNorm2d(in_size),
#             hsigmoid()
#         )
#     def forward(self, x):
#         return x * self.se(x)


#----------------------------------------------------CBAM Module ---------------------------------------------------------#

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()

    def forward(self, x):
        avg_out_ch = self.fc(self.avg_pool(x))
        max_out_ch = self.fc(self.max_pool(x))
        out_ch = avg_out_ch + max_out_ch
        out_ch = self.sigmoid1(out_ch)
        out_ch =  out_ch * x

        avg_out_sp = torch.mean(out_ch, dim=1, keepdim=True)
        max_out_sp, _ = torch.max(out_ch, dim=1, keepdim=True)
        out_sp = torch.cat([avg_out_sp, max_out_sp], dim=1)
        out = self.sigmoid2(self.conv1(out_sp)) * out_ch

        return out



class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)





