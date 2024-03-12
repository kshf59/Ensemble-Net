import torch
import torch.nn as nn
import torch.nn.functional as F

from model.unet.unet_model import UNet
from model.segnet.segnet_model import SegNet
from model.Enet.enet import ENet
#from torchvision.models.segmentation import deeplabv3_resnet101 as DeepLabv3
#from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large as DeepLabv3
#from torchvision.models.segmentation import lraspp_mobilenet_v3_large as DeepLabv3

#from torchvision.models.segmentation import fcn_resnet50 as DeepLabv3

#voting
#stacking
#feature fusion

class ChannelSpatialAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelSpatialAttentionModule, self).__init__()
        reduced_channels = max(1, in_channels // reduction_ratio)

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        #print(f"Input x shape: {x.shape}")  # 입력 크기 출력
        channel_att = self.channel_attention(x)
        #print(f"Channel attention shape: {channel_att.shape}")  # 채널 어텐션 후 크기 출력
        channel_att = x * channel_att
        
        # Spatial attention
        max_pool = torch.max(channel_att, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(channel_att, dim=1, keepdim=True)
        spatial_att_input = torch.cat([max_pool, avg_pool], dim=1)
        spatial_att = self.spatial_attention(spatial_att_input)
        spatial_att = channel_att * spatial_att
        #print(f"Final output shape: {spatial_att.shape}")  # 최종 출력 크기 출력
        
        return spatial_att
    
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
            self.unet = UNet(n_channels=self.n_channels, n_classes=self.n_classes, bilinear=True)
            self.segnet = SegNet(n_channels=self.n_channels, n_classes=self.n_classes)
            self.enet = ENet(self.n_classes)
            
            
            # Replace AttentionModule with ChannelSpatialAttentionModule
            self.attention_unet = ChannelSpatialAttentionModule(self.n_classes)
            self.attention_segnet = ChannelSpatialAttentionModule(self.n_classes)
            self.attention_enet = ChannelSpatialAttentionModule(self.n_classes)
            
            # Additional layers for ensemble
            self.conv1x1_unet = nn.Conv2d(self.n_classes, self.n_classes, kernel_size=1)
            self.conv1x1_segnet = nn.Conv2d(self.n_classes, self.n_classes, kernel_size=1)
            self.conv1x1_enet = nn.Conv2d(self.n_classes, self.n_classes, kernel_size=1)

            # Convolution to concatenate all model outputs
            self.conv_concat = nn.Conv2d(self.n_classes * 3, self.n_classes, kernel_size=1)

            # Enhanced feature extraction layers
            self.feature_enhancement = nn.Sequential(
                nn.Conv2d(self.n_classes, self.n_classes * 4, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.n_classes * 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.n_classes * 4, self.n_classes * 4, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.n_classes * 4),
                nn.ReLU(inplace=True)
            )

            # Final convolution
            self.conv_out = nn.Conv2d(self.n_classes * 4, self.n_classes, kernel_size=1)

    

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
            unet_out = self.attention_unet(self.conv1x1_unet(self.unet(x)))
            segnet_out = self.attention_segnet(self.conv1x1_segnet(self.segnet(x)))
            enet_out = self.attention_enet(self.conv1x1_enet(self.enet(x)))

            # Concatenate all model outputs and apply convolution
            concatenated = torch.cat((unet_out, segnet_out, enet_out), dim=1)
            concatenated = self.conv_concat(concatenated)

            # Enhance features
            enhanced = self.feature_enhancement(concatenated)

            # Generate final output
            out = self.conv_out(enhanced)
            
            
            return out     
            

