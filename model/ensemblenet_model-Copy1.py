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

class AttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.sigmoid(self.conv(x))
        return x * attention


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
            
            # Additional layers for ensemble
            self.conv1x1_unet = nn.Conv2d(self.n_classes, self.n_classes, kernel_size=1)
            self.conv1x1_segnet = nn.Conv2d(self.n_classes, self.n_classes, kernel_size=1)
            self.conv1x1_enet = nn.Conv2d(self.n_classes, self.n_classes, kernel_size=1)

            # Attention modules for each model output
            self.attention_unet = AttentionModule(self.n_classes, self.n_classes)
            self.attention_segnet = AttentionModule(self.n_classes, self.n_classes)
            self.attention_enet = AttentionModule(self.n_classes, self.n_classes)

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

            # Concatenate all model outputs
            concatenated = torch.cat((unet_out, segnet_out, enet_out), dim=1)

            # Apply convolution to concatenated outputs
            concatenated = self.conv_concat(concatenated)

            # Enhance features
            enhanced = self.feature_enhancement(concatenated)

            # Generate final output
            out = self.conv_out(enhanced)
            
            
            return out     
            

