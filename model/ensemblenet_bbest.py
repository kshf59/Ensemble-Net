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

            self.conv_concat = nn.Conv2d(self.n_classes * 3, self.n_classes, kernel_size=1)

            self.conv_batchnorm1 = nn.Sequential(
                nn.Conv2d(self.n_classes, self.n_classes * 4, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.n_classes * 4),
                nn.ReLU(inplace=True)
            )
            self.conv_batchnorm2 = nn.Sequential(
                nn.Conv2d(self.n_classes * 4, self.n_classes * 4, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.n_classes * 4),
                nn.ReLU(inplace=True)
            )

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
            '''
            fused_out = torch.cat([self.unet(x), self.segnet(x), self.enet(x)], dim=1)
            bn_out = self.conv_batchnorm1(fused_out)
            bn_out = self.conv_batchnorm2(bn_out)
            bn_out = self.conv_batchnorm3(bn_out)
            bn_out = self.conv_batchnorm4(bn_out)
            out = self.conv_out(bn_out)
            '''
            
            unet_output = self.unet(x)
            segnet_output = self.segnet(x)
            enet_output = self.enet(x)    
            
            # Apply 1x1 convolutions for dimension reduction
            unet_output = self.conv1x1_unet(unet_output)
            segnet_output = self.conv1x1_segnet(segnet_output)
            enet_output = self.conv1x1_enet(enet_output)

            # Upsample ENet output to match other outputs
            enet_output = F.interpolate(enet_output, size=segnet_output.shape[2:], mode='bilinear', align_corners=True)

            # Concatenate outputs
            concat_output = torch.cat((unet_output, segnet_output, enet_output), dim=1)

            # Additional layers
            concat_output = self.conv_concat(concat_output)
            concat_output = self.conv_batchnorm1(concat_output)
            concat_output = self.conv_batchnorm2(concat_output)

            # Final convolution for output
            out = self.conv_out(concat_output)
        
            
            
            return out     
            

