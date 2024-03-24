import torch
import torch.nn as nn
import torch.nn.functional as F

from model.unet.unet_model import UNet
from model.segnet.segnet_model import SegNet
from model.Enet.enet import ENet

# dilation_rates = [1, 2, 4]
#ensemble_fusion Validation Pixel Accuracy: 0.9723759701377467
#ensemble_fusion Validation MIoU: 0.9025760462802437
#ensemble_fusion Validation Dice Score: 0.9256073236465454

# dilation_rate = [1, 2, 4, 8]
#ensemble_fusion Validation Pixel Accuracy: 0.9795141387404057
#ensemble_fusion Validation MIoU: 0.9250726334209836
#ensemble_fusion Validation Dice Score: 0.9345816969871521


class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.sigmoid(self.conv(x))
        return x * attention



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias = False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias = False)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        return x

'''
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, internal_ratio=4):
        super(Bottleneck, self).__init__()
        internal_channels = in_channels // internal_ratio
        # Reduce phase
        self.conv_reduce = nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=False)
        self.norm_reduce = nn.BatchNorm2d(internal_channels)
        # Transform phase
        self.conv_transform = nn.Conv2d(internal_channels, internal_channels, kernel_size=3, padding=1, bias=False)
        self.norm_transform = nn.BatchNorm2d(internal_channels)
        # Expand phase
        self.conv_expand = nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False)
        self.norm_expand = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.conv_reduce(x)
        x = self.norm_reduce(x)
        x = self.relu(x)
        x = self.conv_transform(x)
        x = self.norm_transform(x)
        x = self.relu(x)
        x = self.conv_expand(x)
        x = self.norm_expand(x)
        x += residual  # Skip Connection
        x = self.relu(x)
        return x
'''

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, internal_ratio=4, dilation_rates=[1, 2, 4, 8, 16]):
        super(Bottleneck, self).__init__()
        internal_channels = in_channels // internal_ratio
        self.dilation_rates = dilation_rates
        
        # Reduce phase
        self.conv_reduce = nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=False)
        self.norm_reduce = nn.BatchNorm2d(internal_channels)
        
        # Transform phase with multiple dilated convolutions
        self.conv_transforms = nn.ModuleList([
            nn.Conv2d(internal_channels, internal_channels, kernel_size=3, 
                      padding=rate, dilation=rate, bias=False) 
            for rate in dilation_rates
        ])
        self.norm_transforms = nn.ModuleList([nn.BatchNorm2d(internal_channels) for _ in dilation_rates])
        
        # Combine phase
        self.conv_combine = nn.Conv2d(internal_channels * len(dilation_rates), out_channels, kernel_size=1, bias=False)
        self.norm_combine = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        
        x = self.conv_reduce(x)
        x = self.norm_reduce(x)
        x = self.relu(x)
        
        # Apply each dilated convolution independently
        transformed_features = [conv(x) for conv in self.conv_transforms]
        
        # Normalize each transformed feature independently
        transformed_features = [self.relu(norm(feat)) for norm, feat in zip(self.norm_transforms, transformed_features)]
        
        # Concatenate along the channel dimension
        x = torch.cat(transformed_features, dim=1)
        
        x = self.conv_combine(x)
        x = self.norm_combine(x)
        x += residual  # Skip Connection
        x = self.relu(x)
        
        return x
    
    

#ensemble_fusion Validation MIoU: 0.891731963875313
#ensemble_fusion Validation Dice Score: 0.9214223027229309
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, use_bottleneck=True):
        super(Encoder, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.use_bottleneck = use_bottleneck
        if self.use_bottleneck:
            self.bottleneck = Bottleneck(out_channels, out_channels) # in_channels와 out_channels는 동일하게 유지
        self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)
        
        self.attention = AttentionModule(out_channels)
        
    def forward(self, x):
        x = self.conv_block(x)
        if self.use_bottleneck:
            x = self.bottleneck(x)
        x = self.attention(x)
        x, indices = self.pool(x)
        return x, indices


'''
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, use_bottleneck=True):
        super(Encoder, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.use_bottleneck = use_bottleneck
        if self.use_bottleneck:
            self.bottleneck = Bottleneck(out_channels, out_channels) # in_channels와 out_channels는 동일하게 유지
        self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)
        
        self.attention = AttentionModule(out_channels)
        self.conv_concat = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x1 = self.conv_block(x)
        
        if self.use_bottleneck:
            x2 = self.bottleneck(x1)
            
        x1 = self.attention(x1)
        x2 = self.attention(x2)
        
        concatenated = torch.cat((x1, x2), dim=1)
        x = self.conv_concat(concatenated)
        x = self.norm(x)
        
        x, indices = self.pool(x)
        return x, indices
'''

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, use_bottleneck=False):
        super(Decoder, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, stride=2)
        self.conv_block = ConvBlock(in_channels, out_channels)
        #self.use_bottleneck = use_bottleneck
        #if self.use_bottleneck:
        #    self.bottleneck = Bottleneck(out_channels, out_channels)
        
    def forward(self, x, indices):
        x = self.unpool(x, indices)
        #if self.use_bottleneck:
        #    x = self.bottleneck(x)
        x = self.conv_block(x)
        return x
    
class EnsembleNet(nn.Module):
    def __init__(self, model_name, n_ch, n_cls):
        super(EnsembleNet, self).__init__()
        
        self.model_name = model_name
        self.n_channels = n_ch
        self.n_classes = n_cls
        
        if 'ensemble' in self.model_name:
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

        # Attention Mechanism
        #x = self.attention1(x, x)  # encoder4의 출력에 Attention 적용
        x = self.decoder4(x, indices4)
        #x = self.attention2(x, x)  # encoder3의 출력에 Attention 적용
        x = self.decoder3(x, indices3)
        #print('dncoder 3 shape X : {}'.format(x.shape))

        #x = self.attention3(x, x)  # encoder2의 출력에 Attention 적용
        x = self.decoder2(x, indices2)

        #x = self.attention4(x, x)  # encoder1의 출력에 Attention 적용
        x = self.decoder1(x, indices1)

        # 최종 출력
        out = self.final_conv(x)
            
        return out