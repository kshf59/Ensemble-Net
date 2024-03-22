import torch
import torch.nn as nn
import torch.nn.functional as F

from model.unet.unet_model import UNet
from model.segnet.segnet_model import SegNet
from model.Enet.enet import ENet

'''
## 이건 사실 residual block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        #self.relu = nn.ReLU(inplace=False)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out
'''

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            #nn.ReLU(inplace=True),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            #nn.ReLU(inplace=True)
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

    
class Encoder(nn.Module):
    def __init__(self, n_channels, channels=(64, 128, 256, 512, 1024)):
        super(Encoder, self).__init__()
        self.enc_blocks = nn.ModuleList([ConvBlock(n_channels if i == 0 else channels[i-1], channels[i]) for i in range(len(channels))])

    def forward(self, x):
        block_outputs = []
        for block in self.enc_blocks:
            x = block(x)
            block_outputs.append(x)
            x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        return block_outputs

class Decoder(nn.Module):
    def __init__(self, n_classes, channels=(1024, 512, 256, 128, 64)):
        super(Decoder, self).__init__()
        # 마지막 채널을 n_classes가 아니라, 중간 채널 수로 변경
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(channels[i], channels[i+1], kernel_size=2, stride=2) for i in range(len(channels)-1)])
        self.dec_blocks = nn.ModuleList([ConvBlock(channels[i], channels[i+1]) for i in range(len(channels)-1)])
        # n_classes에 맞는 출력 레이어는 EnsembleNet 내에서 처리

    def forward(self, x, enc_features):
        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            enc_feat = self.crop(enc_features[i], x)
            x = torch.cat([x, enc_feat], dim=1)
            x = self.dec_blocks[i](x)
        return x

    def crop(self, enc_features, x):
        _, _, H, W = x.size()
        enc_features = F.interpolate(enc_features, size=(H, W), mode='bilinear', align_corners=True)
        return enc_features


class EnsembleNet(nn.Module):
    def __init__(self, model_name, n_ch, n_cls):
        super(EnsembleNet, self).__init__()
        
        self.model_name = model_name
        self.n_channels = n_ch
        self.n_classes = n_cls
        
        if self.model_name.lower() not in ("unet", "segnet", 'enet', "ensemble_voting", "ensemble_fusion"):
            raise ValueError("'model_name' should be one of ('unet', 'segnet', 'enet', 'ensemble_voting', 'ensemble_fusion')")
        
        if self.model_name == 'unet':
            self.unet = UNet(n_channels=self.n_channels, n_classes=self.n_classes, bilinear=True)
            
        if self.model_name == 'segnet':
            self.segnet = SegNet(n_channels=self.n_channels, n_classes=self.n_classes)
            
        if self.model_name == 'enet':
            #self.deeplab = DeepLabv3(num_classes = self.n_classes, pretrained = False)
            self.enet = model = ENet(self.n_classes)
            
        if 'ensemble' in self.model_name:
            # Base models
            self.unet = UNet(n_channels=self.n_channels, n_classes=self.n_classes, bilinear=True)
            self.segnet = SegNet(n_channels=self.n_channels, n_classes=self.n_classes)
            self.enet = ENet(self.n_classes)
            
            self.encoder = Encoder(self.n_channels)
            self.decoder = Decoder(self.n_classes)
            self.head = nn.Conv2d(64, self.n_classes, kernel_size=1)  # 최종 출력 채널을 클래스 수로 설정


    def forward(self, x):
    
        if self.model_name == 'unet':
            out = self.unet(x)
            return out
                
        if self.model_name == 'segnet':
            out = self.segnet(x)
            return out
                
        if self.model_name == 'enet':
            out = self.enet(x)
            return out
                
        if self.model_name == 'ensemble_voting':
            unet_out = self.unet(x)
            segnet_out = self.segnet(x)
            enet_out = self.enet(x)
            return unet_out, segnet_out, enet_out
                
        if self.model_name == 'ensemble_fusion':
                
            enc_features = self.encoder(x)
            
            dec_output = self.decoder(enc_features[::-1][0], enc_features[::-1][1:])
            out = self.head(dec_output)

            return out     
