import torch
import torch.nn as nn
import torch.nn.functional as F
from .ensemble_fusion_parts import *



# dilation_rates = [1, 2, 4]
#ensemble_fusion Validation Pixel Accuracy: 0.9723759701377467
#ensemble_fusion Validation MIoU: 0.9025760462802437
#ensemble_fusion Validation Dice Score: 0.9256073236465454

# dilation_rate = [1, 2, 4, 8]
#ensemble_fusion Validation Pixel Accuracy: 0.9795141387404057
#ensemble_fusion Validation MIoU: 0.9250726334209836
#ensemble_fusion Validation Dice Score: 0.9345816969871521


class EnsembleFusion(nn.Module):
    
    def __init__(self, n_channels, n_classes):
        
        super(EnsembleFusion, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        self.nch = 64

        # Encoder Layers
        self.encoder1 = Encoder(self.n_channels, self.nch, use_bottleneck=True)
        self.encoder2 = Encoder(self.nch, self.nch * 2, use_bottleneck=True)
        self.encoder3 = Encoder(self.nch * 2, self.nch * 4, use_bottleneck=True)
        self.encoder4 = Encoder(self.nch * 4, self.nch * 8, use_bottleneck=True)

        # Decoder Layers
        self.decoder4 = Decoder(self.nch * 8, self.nch * 4, use_bottleneck=True)
        self.decoder3 = Decoder(self.nch * 4, self.nch * 2, use_bottleneck=True)
        self.decoder2 = Decoder(self.nch * 2, self.nch, use_bottleneck=True)
        self.decoder1 = Decoder(self.nch, self.nch, use_bottleneck=True)

        # Final Convolution
        self.final_conv = nn.Conv2d(self.nch, self.n_classes, kernel_size=1)         
            

    def forward(self, x):
        
        # Encoder pathway
        x, indices1 = self.encoder1(x)
        x, indices2 = self.encoder2(x)
        x, indices3 = self.encoder3(x)
        x, indices4 = self.encoder4(x)

        x = self.decoder4(x, indices4)
        x = self.decoder3(x, indices3)
        x = self.decoder2(x, indices2)
        x = self.decoder1(x, indices1)

        # output
        out = self.final_conv(x)
            
        return out