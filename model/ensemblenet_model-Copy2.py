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
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, dilation=dilation, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.aspp1 = DepthwiseSeparableConv(in_channels, out_channels, dilation=1)
        self.aspp2 = DepthwiseSeparableConv(in_channels, out_channels, dilation=6, padding=6)
        self.aspp3 = DepthwiseSeparableConv(in_channels, out_channels, dilation=12, padding=12)
        self.aspp4 = DepthwiseSeparableConv(in_channels, out_channels, dilation=18, padding=18)

        self.conv_out = nn.Conv2d(out_channels*4, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv_out(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



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
            # Base models
            self.unet = UNet(n_channels=self.n_channels, n_classes=self.n_classes, bilinear=True)
            self.segnet = SegNet(n_channels=self.n_channels, n_classes=self.n_classes)
            self.enet = ENet(self.n_classes)

            # Attention modules for each model output
            self.attention_unet = AttentionModule(self.n_classes, self.n_classes)
            self.attention_segnet = AttentionModule(self.n_classes, self.n_classes)
            self.attention_enet = AttentionModule(self.n_classes, self.n_classes)

            # Adaptive Weights for model outputs
            #self.weights = nn.Parameter(torch.ones(3))
            
            
            # Convolution to concatenate all model outputs
            self.conv_concat = nn.Conv2d(self.n_classes * 3, self.n_classes, kernel_size=1)
            
            
            
            # Enhanced feature extraction layer
      
            self.feature_enhancement = nn.Sequential(
                nn.Conv2d(2, self.n_classes * 8, kernel_size=1),
                nn.BatchNorm2d(self.n_classes * 8),
                nn.ReLU(inplace=True),
                #nn.Dropout(0.5),
                ASPP(self.n_classes * 8, self.n_classes * 8),
                #nn.Dropout(0.5),
                DepthwiseSeparableConv(self.n_classes * 8, self.n_classes * 4, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.n_classes * 4),
                nn.ReLU(inplace=True)
            )
            
            # Final convolution to get to the number of classes
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
            # Get outputs from each model
            unet_out = self.attention_unet(self.unet(x))
            segnet_out = self.attention_segnet(self.segnet(x))
            enet_out = self.attention_enet(self.enet(x))
            
            # Concatenate all model outputs
            concatenated = torch.cat((unet_out, segnet_out, enet_out), dim=1)
            # Apply convolution to concatenated outputs
            weighted_sum = self.conv_concat(concatenated)

            
            #unet_out = self.unet(x)
            #segnet_out = self.segnet(x)
            #enet_out = self.enet(x)

            # Applying softmax to weights for normalization
            #normalized_weights = F.softmax(self.weights, dim=0)
            # Weighted sum of model outputs
            #weighted_sum = normalized_weights[0] * unet_out + normalized_weights[1] * segnet_out + normalized_weights[2] * enet_out
           

            #print(weighted_sum.shape)
            # Instead of concatenating, we use the weighted sum directly
            # Enhanced feature extraction applied to the weighted sum of model outputs
            enhanced_features = self.feature_enhancement(weighted_sum)

            # Generate final output
            out = self.conv_out(enhanced_features)
            
            return out     
            

