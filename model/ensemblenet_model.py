import torch
import torch.nn as nn
import torch.nn.functional as F

#from model.unet.unet_model import UNet
#from model.segnet.segnet_model import SegNet




class EnsembleNet(nn.Module):
    def __init__(self, model_unet, model_segnet):
        super(EnsembleNet, self).__init__()
        
        
        self.unet = model_unet
        self.segnet = model_segnet
        self.n_classes = model_unet.n_classes
        self.n_channels = model_unet.n_channels
        
        self.conv_layer1 = nn.Conv2d(self.n_classes, self.n_classes, kernel_size=1)
        self.conv_layer2 = nn.Conv2d(self.n_classes, self.n_classes, kernel_size=1)
        
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        
        unet_out = self.unet(x)
        segnet_out = self.segnet(x)
        
        
        out = torch.mul(unet_out, segnet_out)
        out = self.conv_layer1(out)
        #out = self.relu(out)

        return out




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




