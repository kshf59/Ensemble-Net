import torch
import torch.nn as nn
import torch.nn.functional as F


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


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, internal_ratio=4, dilation_rates=[1, 2, 4, 8]):
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
    
    
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, use_bottleneck=False):
        super(Decoder, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, stride=2)
        self.conv_block = ConvBlock(in_channels, out_channels)
        
    def forward(self, x, indices):
        x = self.unpool(x, indices)
        x = self.conv_block(x)
        return x
    