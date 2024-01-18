import torch
import torch.nn as nn


class Conv_BatchNorm(nn.Module):
    
    def __init__(self, in_channels, out_channels, bn_momentum):
        super().__init__()
        self.conv_batchnorm = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels, momentum=bn_momentum),
                nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv_batchnorm(x)

'''    
class MaxEnDecoder(nn.Module):
    
    def __init__(self, layer_type):
        super().__init__()
        if layer_type == 'encoder':
            self.MaxPool = nn.MaxPool2d(2, stride=2, return_indices=True)

        elif layer_type == 'decoder':
            self.MaxPool = nn.MaxUnpool2d(2, stride=2) 

    def forward(self, x):
        return self.MaxPool(x)
'''    

  
class MaxPoolEncoder(nn.Module):
    
    def __init__(self, value):
        super().__init__()
        self.encoder = nn.MaxPool2d(value, stride=value, return_indices=True)

    def forward(self, x):
        return self.encoder(x)
    
class MaxPooldecoder(nn.Module):
    
    def __init__(self, value):
        super().__init__()
        self.decoder = nn.MaxUnpool2d(value, stride=value) 

    def forward(self, x):
        return self.decoder(x)    
