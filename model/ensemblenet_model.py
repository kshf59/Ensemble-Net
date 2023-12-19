import torch
import torch.nn as nn
import torch.nn.functional as F

from model.unet.unet_model import UNet
from model.segnet.segnet_model import SegNet



class EnsembleNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(EnsembleNet, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        self.unet = UNet(n_channels=self.n_channels, n_classes=self.n_classes, bilinear=True)
        self.segnet = SegNet(n_channels=self.n_channels, n_classes=self.n_classes)
        
        #self.conv1 = nn.Conv2d(self.n_ch, self.n_cl, kernel_size=3, padding=1)
        #self.fusion_layer = self.fusion(self.unet, self.segnet)
    '''
    def voting(self, x):
        pass
        
    def stacking(self, x):
        pass
    '''
    
    def fusion(self, model1, model2):
        
        return model1, model2
        
    def forward(self, x):
        
        unet = self.unet(x)
        segnet = self.unet(x)
        #out1, out2 = self.fusion(unet, segnet)
        #return result1, result2
        return unet, segnet
    




