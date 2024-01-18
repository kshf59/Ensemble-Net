import torch
import torch.nn as nn
import torch.nn.functional as F

from model.unet.unet_model import UNet
from model.segnet.segnet_model import SegNet
from torchvision.models.segmentation import deeplabv3_resnet101 as DeepLabv3


#voting
#stacking
#feature fusion



class EnsembleNet(nn.Module):
    def __init__(self, model_name, n_ch, n_cls):
        super(EnsembleNet, self).__init__()
        
        self.model_name = model_name
        self.n_channels = n_ch
        self.n_classes = n_cls
        
        if self.model_name.lower() not in ("unet", "segnet", 'deeplabv3', "ensemble_voting", "ensemble_fusion"):
            raise ValueError("'model_name' should be one of ('unet', 'segnet', 'deeplabv3', 'ensemble_voting', 'ensemble_fusion')")
        
        if self.model_name == 'unet':
            self.unet = UNet(n_channels=self.n_channels, n_classes=self.n_classes, bilinear=True)
            
        if self.model_name == 'segnet':
            self.segnet = SegNet(n_channels=self.n_channels, n_classes=self.n_classes)
            
        if self.model_name == 'deeplabv3':
            self.deeplab = DeepLabv3(num_classes = self.n_classes, pretrained = False)
        
        if 'ensemble' in self.model_name:
            self.unet = UNet(n_channels=self.n_channels, n_classes=self.n_classes, bilinear=True)
            self.segnet = SegNet(n_channels=self.n_channels, n_classes=self.n_classes)
            self.deeplab = DeepLabv3(num_classes = self.n_classes, pretrained = False)
            
            self.conv_out1 = nn.Conv2d(self.n_classes * 3, self.n_classes, kernel_size=3, padding=1)
            self.conv_out2 = nn.Conv2d(self.n_classes, self.n_classes, kernel_size=3, padding=1)
        
        '''    
        self.double_conv = nn.Sequential(
            nn.Conv2d(self.n_classes, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        '''
        
    
    def forward(self, x):
    
        if self.model_name == 'unet':
            out = self.unet(x)
    
        if self.model_name == 'segnet':
            out = self.segnet(x)
            
        if self.model_name == 'deeplabv3':
            out = self.deeplab(x)
            
        if self.model_name == 'ensemble_voting':
            out = (F.softmax(self.unet(x), dim=1) + F.softmax(self.segnet(x), dim=1) + F.softmax(self.deeplab(x)['out'], dim=1)) / 3.0
            #out = (self.unet(x) + self.segnet(x) + self.deeplab(x)['out']) / 3.0
            
        if self.model_name == 'ensemble_fusion':
            fused_out = torch.cat([self.unet(x), self.segnet(x), self.deeplab(x)['out']], dim=1)
            conv_out = self.conv_out1(fused_out)
            out = self.conv_out2(conv_out)
            
            
        return out
 

