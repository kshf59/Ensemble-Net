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
        
        if self.model_name.lower() not in ("unet", "segnet", 'deeplabv3', "ensemblenet"):
            raise ValueError("'model_name' should be one of ('unet', 'segnet', 'deeplabv3', 'ensemblenet')")
        
        if self.model_name == 'unet':
            self.unet = UNet(n_channels=self.n_channels, n_classes=self.n_classes, bilinear=True)
            
        if self.model_name == 'segnet':
            self.segnet = SegNet(n_channels=self.n_channels, n_classes=self.n_classes)
            
        if self.model_name == 'deeplabv3':
            self.deeplab = DeepLabv3(num_classes = self.n_classes, pretrained = False)
        
        if self.model_name == 'ensemblenet':
            self.unet = UNet(n_channels=self.n_channels, n_classes=self.n_classes, bilinear=True)
            self.segnet = SegNet(n_channels=self.n_channels, n_classes=self.n_classes)
            self.deeplab = DeepLabv3(num_classes = self.n_classes, pretrained = False)
        
        
    
    def forward(self, x):
    
        if self.model_name == 'unet':
            out = self.unet(x)
    
        if self.model_name == 'segnet':
            out = self.segnet(x)
            
        if self.model_name == 'deeplabv3':
            out = self.deeplab(x)
            
        if self.model_name == 'ensemblenet':
            out = (F.softmax(self.unet(x), dim=1) + F.softmax(self.segnet(x), dim=1) + F.softmax(self.deeplab(x)['out'], dim=1)) / 3.0
            
        return out
    
'''
class EnsembleNet(nn.Module):
    def __init__(self, model_unet, model_segnet, model_deeplab, model_name):
        super(EnsembleNet, self).__init__()
        
        
        self.unet = model_unet
        self.segnet = model_segnet
        self.deeplab = model_deeplab
        
        self.n_classes = model_unet.n_classes
        self.n_channels = model_unet.n_channels
        
        self.model_name = model_name
        
        self.conv_layer1 = nn.Conv2d(self.n_classes, self.n_classes, kernel_size=1)
        self.conv_layer2 = nn.Conv2d(self.n_classes, self.n_classes, kernel_size=1)
        
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, self.model_name):
        
        if self.model_name == 'unet':
            unet_out = self.unet(x)
        
        if self.model_name == 'segnet':
            segnet_out = self.segnet(x)
        
        if self.model_name == 'esemblenet':
        
        
        out = torch.mul(unet_out, segnet_out)
        out = self.conv_layer1(out)
        #out = self.relu(out)

        return out

'''


'''
class EnsembleNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(EnsembleNet, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        self.unet = UNet(n_channels=self.n_channels, n_classes=self.n_classes, bilinear=True)
        self.segnet = SegNet(n_channels=self.n_channels, n_classes=self.n_classes)
        
        self.conv_layer = nn.Conv2d(64, self.n_classes, kernel_size=3, padding=1)
        #self.fusion_layer = self.fusion(self.unet, self.segnet)
        
        
    
    def voting(self, x):
        pass
        
    def stacking(self, x):
        pass
    
    def fusion(self, model1, model2):
        
        result = torch.mul(model1, model2)
        
        return result
    
    
    def forward(self, x):
        
        unet = self.unet(x)
        #segnet = self.segnet(x)
        
        
        #out = torch.mul(unet, segnet)
        
        #out = self.conv_layer(fusion)

        return unet
'''




