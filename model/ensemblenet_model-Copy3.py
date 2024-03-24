import torch
import torch.nn as nn
import torch.nn.functional as F

from model.unet.unet_model import UNet
from model.segnet.segnet_model import SegNet
from model.Enet.enet import ENet

## 현재 1등 그냥 컨볼루션, batch 노말라이제이션, bias=False  d91.5 m88
## 현재 1등 그냥 컨볼루션, instance 노말라이제이션, bias=True  d91 m86
## 현재 2등 그냥 컨볼루션, 그룹 노말라이제이션  bias=False d90 m84


class Bottleneck(nn.Module):
    """Regular bottlenecks are the main building block of ENet.
    Main branch:
    1. Shortcut connection.

    Extension branch:
    1. 1x1 convolution which decreases the number of channels by
    ``internal_ratio``, also called a projection;
    2. regular, dilated or asymmetric convolution;
    3. 1x1 convolution which increases the number of channels back to
    ``channels``, also called an expansion;
    4. dropout as a regularizer.

    Keyword arguments:
    - channels (int): the number of input and output channels.
    - internal_ratio (int, optional): a scale factor applied to
    ``channels`` used to compute the number of
    channels after the projection. eg. given ``channels`` equal to 128 and
    internal_ratio equal to 2 the number of channels after the projection
    is 64. Default: 4.
    - kernel_size (int, optional): the kernel size of the filters used in
    the convolution layer described above in item 2 of the extension
    branch. Default: 3.
    - padding (int, optional): zero-padding added to both sides of the
    input. Default: 0.
    - dilation (int, optional): spacing between kernel elements for the
    convolution described in item 2 of the extension branch. Default: 1.
    asymmetric (bool, optional): flags if the convolution described in
    item 2 of the extension branch is asymmetric or not. Default: False.
    - dropout_prob (float, optional): probability of an element to be
    zeroed. Default: 0 (no dropout).
    - bias (bool, optional): Adds a learnable bias to the output if
    ``True``. Default: False.
    - relu (bool, optional): When ``True`` ReLU is used as the activation
    function; otherwise, PReLU is used. Default: True.

    """

    def __init__(self,
                 channels,
                 internal_ratio=4,
                 kernel_size=3,
                 padding=0,
                 dilation=1,
                 asymmetric=False,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        super().__init__()

        # Check in the internal_scale parameter is within the expected range
        # [1, channels]
        if internal_ratio <= 1 or internal_ratio > channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}."
                               .format(channels, internal_ratio))

        internal_channels = channels // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        # Main branch - shortcut connection

        # Extension branch - 1x1 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution, and,
        # finally, a regularizer (spatial dropout). Number of channels is constant.

        # 1x1 projection convolution
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                channels,
                internal_channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(internal_channels), activation())

        # If the convolution is asymmetric we split the main convolution in
        # two. Eg. for a 5x5 asymmetric convolution we have two convolution:
        # the first is 5x1 and the second is 1x5.
        if asymmetric:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(kernel_size, 1),
                    stride=1,
                    padding=(padding, 0),
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation(),
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(1, kernel_size),
                    stride=1,
                    padding=(0, padding),
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation())
        else:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation())

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(channels), activation())

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after adding the branches
        self.out_activation = activation()

    def forward(self, x):
        # Main branch shortcut
        main = x

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Add main and extension branches
        out = main + ext

        return self.out_activation(out)



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalization='batch', enderelu = False):
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias = False)
        if normalization == 'batch':
            self.norm1 = nn.BatchNorm2d(out_channels)
        elif normalization == 'instance':
            self.norm1 = nn.InstanceNorm2d(out_channels)
        elif normalization == 'layer':
            self.norm1 = nn.GroupNorm(32, out_channels)  # LayerNorm is equivalent to GroupNorm with num_groups=1
        self.relu1 = nn.ReLU(inplace=True)
        
    
        '''
        self.bottleneck1 = Bottleneck(out_channels, padding=1, dropout_prob=0.1, relu=enderelu)
        self.bottleneck2 = Bottleneck(out_channels, dilation=2, padding=2, dropout_prob=0.1, relu=enderelu)
        self.asybottleneck3 = Bottleneck(out_channels,
        kernel_size=5,
        padding=2,
        asymmetric=True,
        dropout_prob=0.1,
        relu=enderelu)
        self.bottleneck4 = Bottleneck(out_channels, dilation=4, padding=4, dropout_prob=0.1, relu=enderelu)
        self.bottleneck5 = Bottleneck(out_channels, padding=1, dropout_prob=0.1, relu=enderelu)
        self.bottleneck6 = Bottleneck(out_channels, dilation=8, padding=8, dropout_prob=0.1, relu=enderelu)
        self.asybottleneck7 = Bottleneck(out_channels,
        kernel_size=5,
        asymmetric=True,
        padding=2,
        dropout_prob=0.1,
        relu=enderelu)
        self.bottleneck8 = Bottleneck(out_channels, dilation=16, padding=16, dropout_prob=0.1, relu=enderelu)        
        '''
        
        
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias = False)
        if normalization == 'batch':
            self.norm2 = nn.BatchNorm2d(out_channels)
        elif normalization == 'instance':
            self.norm2 = nn.InstanceNorm2d(out_channels)
        elif normalization == 'layer':
            self.norm2 = nn.GroupNorm(32, out_channels)  # LayerNorm is equivalent to GroupNorm with num_groups=1
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        
        '''
        x=self.bottleneck1(x)
        x=self.bottleneck2(x)
        x=self.asybottleneck3(x)
        x=self.bottleneck4(x)
        x=self.bottleneck5(x)
        x=self.bottleneck6(x)
        x=self.asybottleneck7(x)
        x=self.bottleneck8(x)
        '''
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        
        return x
    
    
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels, normalization='batch', enderelu = False)
        self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)
        
    def forward(self, x):
        x = self.conv_block(x)
        x, indices = self.pool(x)
        
        return x, indices

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, stride=2)
        self.conv_block = ConvBlock(in_channels, out_channels, normalization='batch', enderelu = True)
        
    
    def forward(self, x, indices):
        
        x = self.unpool(x, indices)
        x = self.conv_block(x)
        
        
        return x

class EnsembleNet(nn.Module):
    def __init__(self, model_name, n_ch, n_cls):
        super(EnsembleNet, self).__init__()
        
        self.model_name = model_name
        self.n_channels = n_ch
        self.n_classes = n_cls
        

        if 'ensemble' in self.model_name:

            self.nch = 32
            
            self.encoder1 = Encoder(self.n_channels, self.nch)
            self.encoder2 = Encoder(self.nch, self.nch * 2)
            self.encoder3 = Encoder(self.nch * 2, self.nch * 4)
            self.encoder4 = Encoder(self.nch * 4, self.nch * 8)

            self.decoder4 = Decoder(self.nch * 8, self.nch * 4)
            self.decoder3 = Decoder(self.nch * 4, self.nch * 2)
            self.decoder2 = Decoder(self.nch * 2, self.nch)
            self.decoder1 = Decoder(self.nch, self.nch)

            self.final_conv = nn.Conv2d(self.nch, self.n_classes, kernel_size=1)            


    def forward(self, x):
    
                
        if self.model_name == 'ensemble_fusion':

            # Encoder pathway
            x, indices1 = self.encoder1(x)
            x, indices2 = self.encoder2(x)
            x, indices3 = self.encoder3(x)
            x, indices4 = self.encoder4(x)


            x = self.decoder4(x, indices4)
            x = self.decoder3(x, indices3)
            x = self.decoder2(x, indices2)
            x = self.decoder1(x, indices1)
            
            # 최종 출력
            out = self.final_conv(x)
            
            return out     



