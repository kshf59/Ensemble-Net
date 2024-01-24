import torch
import torch.nn as nn
import torch.nn.functional as F
from .segnet_parts import *




class SegNet(nn.Module):

    def __init__(self, n_channels, n_classes, BN_momentum=0.5):
        super(SegNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.out_channels = [64, 128, 256, 512]
              
        self.maxpool_en = nn.MaxPool2d(2, stride=2, return_indices=True)
        
        self.convbn_en11 = Conv_BatchNorm(self.n_channels, self.out_channels[0], BN_momentum)
        self.convbn_en12 = Conv_BatchNorm(self.out_channels[0], self.out_channels[0], BN_momentum)
        
        self.convbn_en21 = Conv_BatchNorm(self.out_channels[0], self.out_channels[1], BN_momentum)
        self.convbn_en22 = Conv_BatchNorm(self.out_channels[1], self.out_channels[1], BN_momentum)

        self.convbn_en31 = Conv_BatchNorm(self.out_channels[1], self.out_channels[2], BN_momentum)
        self.convbn_en32 = Conv_BatchNorm(self.out_channels[2], self.out_channels[2], BN_momentum)
        self.convbn_en33 = Conv_BatchNorm(self.out_channels[2], self.out_channels[2], BN_momentum)  
        
        self.convbn_en41 = Conv_BatchNorm(self.out_channels[2], self.out_channels[3], BN_momentum)
        self.convbn_en42 = Conv_BatchNorm(self.out_channels[3], self.out_channels[3], BN_momentum)
        self.convbn_en43 = Conv_BatchNorm(self.out_channels[3], self.out_channels[3], BN_momentum)
        
        self.convbn_en51 = Conv_BatchNorm(self.out_channels[3], self.out_channels[3], BN_momentum)
        self.convbn_en52 = Conv_BatchNorm(self.out_channels[3], self.out_channels[3], BN_momentum)            
        self.convbn_en53 = Conv_BatchNorm(self.out_channels[3], self.out_channels[3], BN_momentum)  
        
        self.maxpool_de = nn.MaxUnpool2d(2, stride=2) 
               
        self.convbn_de53 = Conv_BatchNorm(self.out_channels[3], self.out_channels[3], BN_momentum)
        self.convbn_de52 = Conv_BatchNorm(self.out_channels[3], self.out_channels[3], BN_momentum)
        self.convbn_de51 = Conv_BatchNorm(self.out_channels[3], self.out_channels[3], BN_momentum)        
        
        
        self.convbn_de43 = Conv_BatchNorm(self.out_channels[3], self.out_channels[3], BN_momentum)
        self.convbn_de42 = Conv_BatchNorm(self.out_channels[3], self.out_channels[3], BN_momentum)
        self.convbn_de41 = Conv_BatchNorm(self.out_channels[3], self.out_channels[2], BN_momentum)   

        self.convbn_de33 = Conv_BatchNorm(self.out_channels[2], self.out_channels[2], BN_momentum)
        self.convbn_de32 = Conv_BatchNorm(self.out_channels[2], self.out_channels[2], BN_momentum)
        self.convbn_de31 = Conv_BatchNorm(self.out_channels[2], self.out_channels[1], BN_momentum)         
        
        self.convbn_de22 = Conv_BatchNorm(self.out_channels[1], self.out_channels[1], BN_momentum)
        self.convbn_de21 = Conv_BatchNorm(self.out_channels[1], self.out_channels[0], BN_momentum)
        
        
        self.convbn_de12 = Conv_BatchNorm(self.out_channels[0], self.out_channels[0], BN_momentum) 
        self.conv_de11 = nn.Conv2d(self.out_channels[0], self.n_classes, kernel_size=3, padding=1)

        
    def forward(self, x):

        # ENCODE LAYERS
        
        # Stage 1
        enx1 = self.convbn_en11(x)
        enx1 = self.convbn_en12(enx1)
        enx1, ind1 = self.maxpool_en(enx1)
        
        # Stage 2
        enx2 = self.convbn_en21(enx1)
        enx2 = self.convbn_en22(enx2)
        enx2, ind2 = self.maxpool_en(enx2)

        
        # Stage 3
        enx3 = self.convbn_en31(enx2)
        enx3 = self.convbn_en32(enx3)
        enx3 = self.convbn_en33(enx3)
        enx3, ind3 = self.maxpool_en(enx3)

        # Stage 4
        enx4 = self.convbn_en41(enx3)
        enx4 = self.convbn_en42(enx4)
        enx4 = self.convbn_en43(enx4)
        enx4, ind4 = self.maxpool_en(enx4)
        
        # Stage 5
        enx5 = self.convbn_en51(enx4)
        enx5 = self.convbn_en52(enx5)
        enx5 = self.convbn_en53(enx5)
        enx5, ind5 = self.maxpool_en(enx5)
          
        
        # DECODER LAYERS
        
        # Stage 5
        dex5 = self.maxpool_de(enx5, ind5, output_size = enx4.size())
        dex5 = self.convbn_de53(dex5)
        dex5 = self.convbn_de52(dex5)
        dex5 = self.convbn_de51(dex5)
        
        # Stage 4
        dex4 = self.maxpool_de(dex5, ind4, output_size = enx3.size())
        dex4 = self.convbn_de43(dex4)
        dex4 = self.convbn_de42(dex4)
        dex4 = self.convbn_de41(dex4)        
        
        # Stage 3
        dex3 = self.maxpool_de(dex4, ind3, output_size = enx2.size())
        dex3 = self.convbn_de33(dex3)
        dex3 = self.convbn_de32(dex3)
        dex3 = self.convbn_de31(dex3)        
        
        # Stage 2
        dex2 = self.maxpool_de(dex3, ind2, output_size = enx1.size())
        dex2 = self.convbn_de22(dex2)
        dex2 = self.convbn_de21(dex2)            
        
        # Stage 1
        dex1 = self.maxpool_de(dex2, ind1)
        dex1 = self.convbn_de12(dex1)
        output = self.conv_de11(dex1)

        return output




'''
class SegNet(nn.Module):
    def __init__(self, n_channels, n_classes, BN_momentum=0.5):
        super(SegNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.out_channels = [64, 128, 256, 512]

        self.encoder_layers = self._make_layers(Conv_BatchNorm, self.n_channels, self.out_channels, BN_momentum)
        self.decoder_layers = self._make_layers(Conv_BatchNorm, self.out_channels[-1], self.out_channels[::-1], BN_momentum)

        self.conv_de11 = nn.Conv2d(self.out_channels[0], self.n_classes, kernel_size=3, padding=1)

    def _make_layers(self, block, in_channels, channels, BN_momentum):
        layers = []
        for out_channels in channels:
            layers += [block(in_channels, out_channels, BN_momentum),
                       block(out_channels, out_channels, BN_momentum)]
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        indices = []
        sizes = []
        for layer in self.encoder_layers:
            if isinstance(layer, nn.MaxPool2d):
                x, idx = layer(x)
                indices.append(idx)
                sizes.append(x.size())

        for layer in self.decoder_layers:
            if isinstance(layer, nn.MaxUnpool2d):
                idx = indices.pop()
                size = sizes.pop()
                x = layer(x, idx, output_size=size)
            else:
                x = layer(x)

        x = self.conv_de11(x)
        return x
        
'''