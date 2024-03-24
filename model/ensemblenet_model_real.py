import torch
import torch.nn as nn
import torch.nn.functional as F

from model.unet.unet_model import UNet
from model.segnet.segnet_model import SegNet
from model.Enet.enet import ENet

## 현재 1등 그냥 컨볼루션, batch 노말라이제이션, bias=False  d91.5 m88
## 현재 1등 그냥 컨볼루션, instance 노말라이제이션, bias=True  d91 m86
## 현재 2등 그냥 컨볼루션, 그룹 노말라이제이션  bias=False d90 m84

'''
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalization='batch'):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        if normalization == 'batch':
            self.norm1 = nn.BatchNorm2d(out_channels)
        elif normalization == 'instance':
            self.norm1 = nn.InstanceNorm2d(out_channels)
        elif normalization == 'layer':
            self.norm1 = nn.GroupNorm(1, out_channels)  # LayerNorm is equivalent to GroupNorm with num_groups=1
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if normalization == 'batch':
            self.norm2 = nn.BatchNorm2d(out_channels)
        elif normalization == 'instance':
            self.norm2 = nn.InstanceNorm2d(out_channels)
        elif normalization == 'layer':
            self.norm2 = nn.GroupNorm(1, out_channels)  # LayerNorm is equivalent to GroupNorm with num_groups=1
        self.relu2 = nn.ReLU(inplace=True)
        
        self.residual_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            ) if in_channels != out_channels else None

    
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        
        if self.residual_block:
            residual = self.residual_block(residual)      
        return F.relu(x + residual)
'''
'''
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalization='batch'):
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias = False)
        if normalization == 'batch':
            self.norm1 = nn.BatchNorm2d(out_channels)
        elif normalization == 'instance':
            self.norm1 = nn.InstanceNorm2d(out_channels)
        elif normalization == 'layer':
            self.norm1 = nn.GroupNorm(32, out_channels)  # LayerNorm is equivalent to GroupNorm with num_groups=1
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias = False)
        if normalization == 'batch':
            self.norm2 = nn.BatchNorm2d(out_channels)
        elif normalization == 'instance':
            self.norm2 = nn.InstanceNorm2d(out_channels)
        elif normalization == 'layer':
            self.norm2 = nn.GroupNorm(32, out_channels)  # LayerNorm is equivalent to GroupNorm with num_groups=1
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        
        return x
'''


'''
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalization='batch', stride=1, expansion=4):
        super(ConvBlock, self).__init__()
        self.expansion = expansion
        mid_channels = out_channels // expansion
        #self.attention = ComprehensiveAttention(out_channels)
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        if normalization == 'batch':
            self.bn1 = nn.BatchNorm2d(mid_channels)
        elif normalization == 'instance':
            self.bn1 = nn.InstanceNorm2d(mid_channels)
        elif normalization == 'layer':
            self.bn1 = nn.GroupNorm(8, mid_channels)  # LayerNorm is equivalent to GroupNorm with num_groups=1
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        if normalization == 'batch':
            self.bn2 = nn.BatchNorm2d(mid_channels)
        elif normalization == 'instance':
            self.bn2 = nn.InstanceNorm2d(mid_channels)
        elif normalization == 'layer':
            self.bn2 = nn.GroupNorm(8, mid_channels)  # LayerNorm is equivalent to GroupNorm with num_groups=1
        
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        if normalization == 'batch':
            self.bn3 = nn.BatchNorm2d(out_channels)
        elif normalization == 'instance':
            self.bn3 = nn.InstanceNorm2d(out_channels)
        elif normalization == 'layer':
            self.bn3 = nn.GroupNorm(8, out_channels)  # LayerNorm is equivalent to GroupNorm with num_groups=1
            
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        #out = self.attention(out)

        out += identity
        out = self.relu(out)

        return out

'''

class SqueezeExcitation(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalization='batch', stride=1, expansion=4):
        super(ConvBlock, self).__init__()
        self.expansion = expansion
        mid_channels = out_channels // expansion
        self.attention = ComprehensiveAttention(out_channels)

        # 1x1 conv
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        # Depthwise 3x3 conv
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        # Squeeze and Excitation block
        self.se = SqueezeExcitation(mid_channels)
        
        # 1x1 conv
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.se(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.attention(out)
        
        out += identity
        out = self.relu(out)

        return out


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels, normalization='batch')
        self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)
        
    def forward(self, x):
        x = self.conv_block(x)
        x, indices = self.pool(x)
        
        return x, indices

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, stride=2)
        self.conv_block = ConvBlock(in_channels, out_channels, normalization='batch')
        
    
    def forward(self, x, indices):
        
        x = self.unpool(x, indices)
        x = self.conv_block(x)
        
        
        return x



class EnsembleNet(nn.Module):
    def __init__(self, model_name, n_ch, n_cls):
        super(EnsembleNet, self).__init__()
        
        self.model_name = model_name
        self.n_channels = n_ch
        self.n_classes = n_cls
        

        if 'ensemble' in self.model_name:

            self.nch = 64
            
            self.encoder1 = Encoder(self.n_channels, self.nch)
            self.encoder2 = Encoder(self.nch, self.nch * 2)
            self.encoder3 = Encoder(self.nch * 2, self.nch * 4)
            self.encoder4 = Encoder(self.nch * 4, self.nch * 8)

            # Attention Blocks 추가
            #self.attention1 = AttentionBlock(self.nch * 8, self.nch * 8, self.nch * 4)  # encoder4와 decoder4 사이
            #self.attention2 = AttentionBlock(self.nch * 4, self.nch * 4, self.nch * 2)   # encoder3와 decoder3 사이
            #self.attention3 = AttentionBlock(self.nch * 2, self.nch * 2, self.nch)     # encoder2와 decoder2 사이
            #self.attention4 = AttentionBlock(self.nch, self.nch, self.nch)     # encoder1와 decoder1 사이

            self.decoder4 = Decoder(self.nch * 8, self.nch * 4)
            self.decoder3 = Decoder(self.nch * 4, self.nch * 2)
            self.decoder2 = Decoder(self.nch * 2, self.nch)
            self.decoder1 = Decoder(self.nch, self.nch)

            self.final_conv = nn.Conv2d(self.nch, self.n_classes, kernel_size=1)            


    def forward(self, x):
    
                
        if self.model_name == 'ensemble_fusion':

            # Encoder pathway
            x, indices1 = self.encoder1(x)
            x, indices2 = self.encoder2(x)
            x, indices3 = self.encoder3(x)
            x, indices4 = self.encoder4(x)

            # Attention Mechanism
            #x = self.attention1(x, x)  # encoder4의 출력에 Attention 적용
            x = self.decoder4(x, indices4)

            #x = self.attention2(x, x)  # encoder3의 출력에 Attention 적용
            x = self.decoder3(x, indices3)

            #x = self.attention3(x, x)  # encoder2의 출력에 Attention 적용
            x = self.decoder2(x, indices2)

            #x = self.attention4(x, x)  # encoder1의 출력에 Attention 적용
            x = self.decoder1(x, indices1)
            
            # 최종 출력
            out = self.final_conv(x)
            
            return out     



