import torch
import torch.nn as nn
import torch.nn.functional as F

from model.unet.unet_model import UNet
from model.segnet.segnet_model import SegNet
from model.Enet.enet import ENet

## 현재 1등 그냥 컨볼루션, batch 노말라이제이션, bias=False  d91.5 m88
## 현재 1등 그냥 컨볼루션, instance 노말라이제이션, bias=True  d91 m86
## 현재 2등 그냥 컨볼루션, 그룹 노말라이제이션  bias=False d90 m84
'''
class InitialBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InitialBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x
'''

class InitialBlock(nn.Module):
    """The initial block is composed of two branches:
    1. a main branch which performs a regular convolution with stride 2;
    2. an extension branch which performs max-pooling.

    Doing both operations in parallel and concatenating their results
    allows for efficient downsampling and expansion. The main branch
    outputs 13 feature maps while the extension branch outputs 3, for a
    total of 16 feature maps after concatenation.

    Keyword arguments:
    - in_channels (int): the number of input channels.
    - out_channels (int): the number output channels.
    - kernel_size (int, optional): the kernel size of the filters used in
    the convolution layer. Default: 3.
    - padding (int, optional): zero-padding added to both sides of the
    input. Default: 0.
    - bias (bool, optional): Adds a learnable bias to the output if
    ``True``. Default: False.
    - relu (bool, optional): When ``True`` ReLU is used as the activation
    function; otherwise, PReLU is used. Default: True.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=False,
                 relu=True):
        super().__init__()

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        # Main branch - As stated above the number of output channels for this
        # branch is the total minus 3, since the remaining channels come from
        # the extension branch
        self.main_branch = nn.Conv2d(
            in_channels,
            out_channels - 3,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=bias)

        # Extension branch
        self.ext_branch = nn.MaxPool2d(3, stride=2, padding=1)

        # Initialize batch normalization to be used after concatenation
        self.batch_norm = nn.BatchNorm2d(out_channels)

        # PReLU layer to apply after concatenating the branches
        self.out_activation = activation()

    def forward(self, x):
        main = self.main_branch(x)
        ext = self.ext_branch(x)

        # Concatenate branches
        out = torch.cat((main, ext), 1)

        # Apply batch normalization
        out = self.batch_norm(out)

        return self.out_activation(out)


class DownsamplingBottleneck(nn.Module):
    """Downsampling bottlenecks further downsample the feature map size.

    Main branch:
    1. max pooling with stride 2; indices are saved to be used for
    unpooling later.

    Extension branch:
    1. 2x2 convolution with stride 2 that decreases the number of channels
    by ``internal_ratio``, also called a projection;
    2. regular convolution (by default, 3x3);
    3. 1x1 convolution which increases the number of channels to
    ``out_channels``, also called an expansion;
    4. dropout as a regularizer.

    Keyword arguments:
    - in_channels (int): the number of input channels.
    - out_channels (int): the number of output channels.
    - internal_ratio (int, optional): a scale factor applied to ``channels``
    used to compute the number of channels after the projection. eg. given
    ``channels`` equal to 128 and internal_ratio equal to 2 the number of
    channels after the projection is 64. Default: 4.
    - return_indices (bool, optional):  if ``True``, will return the max
    indices along with the outputs. Useful when unpooling later.
    - dropout_prob (float, optional): probability of an element to be
    zeroed. Default: 0 (no dropout).
    - bias (bool, optional): Adds a learnable bias to the output if
    ``True``. Default: False.
    - relu (bool, optional): When ``True`` ReLU is used as the activation
    function; otherwise, PReLU is used. Default: True.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 internal_ratio=4,
                 return_indices=False,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        super().__init__()

        # Store parameters that are needed later
        self.return_indices = return_indices

        # Check in the internal_scale parameter is within the expected range
        # [1, channels]
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}. "
                               .format(in_channels, internal_ratio))

        internal_channels = in_channels // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        # Main branch - max pooling followed by feature map (channels) padding
        self.main_max1 = nn.MaxPool2d(
            2,
            stride=2,
            return_indices=return_indices)

        # Extension branch - 2x2 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution. Number
        # of channels is doubled.

        # 2x2 projection convolution with stride 2
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                internal_channels,
                kernel_size=2,
                stride=2,
                bias=bias), nn.BatchNorm2d(internal_channels), activation())

        # Convolution
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                internal_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias), nn.BatchNorm2d(internal_channels), activation())

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(out_channels), activation())

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after concatenating the branches
        self.out_activation = activation()

    def forward(self, x):
        # Main branch shortcut
        if self.return_indices:
            main, max_indices = self.main_max1(x)
        else:
            main = self.main_max1(x)

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Main branch channel padding
        n, ch_ext, h, w = ext.size()
        ch_main = main.size()[1]
        padding = torch.zeros(n, ch_ext - ch_main, h, w)

        # Before concatenating, check if main is on the CPU or GPU and
        # convert padding accordingly
        if main.is_cuda:
            padding = padding.cuda()

        # Concatenate
        main = torch.cat((main, padding), 1)

        # Add main and extension branches
        out = main + ext

        return self.out_activation(out), max_indices
    


class UpsamplingBottleneck(nn.Module):
    """The upsampling bottlenecks upsample the feature map resolution using max
    pooling indices stored from the corresponding downsampling bottleneck.

    Main branch:
    1. 1x1 convolution with stride 1 that decreases the number of channels by
    ``internal_ratio``, also called a projection;
    2. max unpool layer using the max pool indices from the corresponding
    downsampling max pool layer.

    Extension branch:
    1. 1x1 convolution with stride 1 that decreases the number of channels by
    ``internal_ratio``, also called a projection;
    2. transposed convolution (by default, 3x3);
    3. 1x1 convolution which increases the number of channels to
    ``out_channels``, also called an expansion;
    4. dropout as a regularizer.

    Keyword arguments:
    - in_channels (int): the number of input channels.
    - out_channels (int): the number of output channels.
    - internal_ratio (int, optional): a scale factor applied to ``in_channels``
     used to compute the number of channels after the projection. eg. given
     ``in_channels`` equal to 128 and ``internal_ratio`` equal to 2 the number
     of channels after the projection is 64. Default: 4.
    - dropout_prob (float, optional): probability of an element to be zeroed.
    Default: 0 (no dropout).
    - bias (bool, optional): Adds a learnable bias to the output if ``True``.
    Default: False.
    - relu (bool, optional): When ``True`` ReLU is used as the activation
    function; otherwise, PReLU is used. Default: True.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 internal_ratio=4,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        super().__init__()

        # Check in the internal_scale parameter is within the expected range
        # [1, channels]
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}. "
                               .format(in_channels, internal_ratio))

        internal_channels = in_channels // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        # Main branch - max pooling followed by feature map (channels) padding
        self.main_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels))

        # Remember that the stride is the same as the kernel_size, just like
        # the max pooling layers
        self.main_unpool1 = nn.MaxUnpool2d(kernel_size=2)

        # Extension branch - 1x1 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution. Number
        # of channels is doubled.

        # 1x1 projection convolution with stride 1
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, internal_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(internal_channels), activation())

        # Transposed convolution
        self.ext_tconv1 = nn.ConvTranspose2d(
            internal_channels,
            internal_channels,
            kernel_size=2,
            stride=2,
            bias=bias)
        self.ext_tconv1_bnorm = nn.BatchNorm2d(internal_channels)
        self.ext_tconv1_activation = activation()

        # 1x1 expansion convolution
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(
                internal_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels))

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after concatenating the branches
        self.out_activation = activation()

    def forward(self, x, max_indices, output_size):
        # Main branch shortcut
        main = self.main_conv1(x)
        main = self.main_unpool1(
            main, max_indices, output_size=output_size)

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_tconv1(ext, output_size=output_size)
        ext = self.ext_tconv1_bnorm(ext)
        ext = self.ext_tconv1_activation(ext)
        ext = self.ext_conv2(ext)
        ext = self.ext_regul(ext)

        # Add main and extension branches
        out = main + ext

        return self.out_activation(out)
    
    
    
'''
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, asymmetric=False, dropout_prob=0.1):
        super(Bottleneck, self).__init__()

        # 1x1 Convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        # Asymmetric Convolution or Normal Convolution
        if asymmetric:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(5, 1), stride=1, padding=(2, 0), dilation=dilation)
            self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 5), stride=1, padding=(0, 2), dilation=dilation)
        else:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation)
            self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation)

        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.batchnorm3 = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout2d(p=dropout_prob)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        # First 1x1 Convolution
        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = self.relu(out)

        # Second Convolution (Asymmetric or Normal)
        out = self.conv2(out)
        out = self.batchnorm2(out)
        out = self.relu(out)

        # Third Convolution
        out = self.conv3(out)
        out = self.batchnorm3(out)

        # Apply dropout
        out = self.dropout(out)

        # Residual connection
        out += residual

        # Apply ReLU
        out = self.relu(out)

        return out
'''

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



class EnsembleNet(nn.Module):
    def __init__(self, model_name, n_ch, n_cls, encoder_relu=False, decoder_relu=True):
        super(EnsembleNet, self).__init__()
        
        self.model_name = model_name
        self.n_channels = n_ch
        self.n_classes = n_cls
        

        if 'ensemble' in self.model_name:

            self.nch = 64
            self.nch_val = int(self.nch / 4)
            
            # Initial Block
            self.initial_block = InitialBlock(self.n_channels, out_channels=self.nch_val)
            # Stage 1 - Encoder
            self.downsample1_0 = DownsamplingBottleneck(
                self.nch_val,
                self.nch,
                return_indices=True,
                dropout_prob=0.01,
                relu=encoder_relu)
            #self.bottleneck1_0 = Bottleneck(in_channels=16, out_channels=64, dropout_prob=0.01)
            self.bottleneck1_1 = Bottleneck(self.nch, padding=1, dropout_prob=0.01, relu=encoder_relu)
            self.bottleneck1_2 = Bottleneck(self.nch, padding=1, dropout_prob=0.01, relu=encoder_relu)
            self.bottleneck1_3 = Bottleneck(self.nch, padding=1, dropout_prob=0.01, relu=encoder_relu)
            self.bottleneck1_4 = Bottleneck(self.nch, padding=1, dropout_prob=0.01, relu=encoder_relu)

            # Stage 2 - Encoder
            #self.bottleneck2_0 = Bottleneck(in_channels=64, out_channels=128,  asymmetric=True, dropout_prob=0.1)
            self.downsample2_0 = DownsamplingBottleneck(
                self.nch,
                self.nch * 2,
                return_indices=True,
                dropout_prob=0.1,
                relu=encoder_relu)
            self.bottleneck2_1 = Bottleneck(self.nch * 2, padding=1, dropout_prob=0.1, relu=encoder_relu)
            self.bottleneck2_2 = Bottleneck(self.nch * 2, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
            self.asybottleneck2_3 = Bottleneck(self.nch * 2,
            kernel_size=5,
            padding=2,
            asymmetric=True,
            dropout_prob=0.1,
            relu=encoder_relu)
            self.bottleneck2_4 = Bottleneck(self.nch * 2, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
            self.bottleneck2_5 = Bottleneck(self.nch * 2, padding=1, dropout_prob=0.1, relu=encoder_relu)
            self.bottleneck2_6 = Bottleneck(self.nch * 2, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
            self.asybottleneck2_7 = Bottleneck(self.nch * 2,
            kernel_size=5,
            asymmetric=True,
            padding=2,
            dropout_prob=0.1,
            relu=encoder_relu)
            self.bottleneck2_8 = Bottleneck(self.nch * 2, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)

            # Stage 3 - Encoder
            self.bottleneck3_1 = Bottleneck(self.nch * 2, padding=1, dropout_prob=0.1, relu=encoder_relu)
            self.bottleneck3_2 = Bottleneck(self.nch * 2, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
            self.asybottleneck3_3 = Bottleneck(self.nch * 2,
            kernel_size=5,
            padding=2,
            asymmetric=True,
            dropout_prob=0.1,
            relu=encoder_relu)
            self.bottleneck3_4 = Bottleneck(self.nch * 2, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
            self.bottleneck3_5 = Bottleneck(self.nch * 2, padding=1, dropout_prob=0.1, relu=encoder_relu)
            self.bottleneck3_6 = Bottleneck(self.nch * 2, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
            self.asybottleneck3_7 = Bottleneck(self.nch * 2,
            kernel_size=5,
            asymmetric=True,
            padding=2,
            dropout_prob=0.1,
            relu=encoder_relu)
            self.bottleneck3_8 = Bottleneck(self.nch * 2, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)
            
            # Stage 4 - Decoder
            self.upsample4_0 = UpsamplingBottleneck(self.nch * 2, self.nch, dropout_prob=0.1, relu=decoder_relu)
            self.bottleneck4_1 = Bottleneck(self.nch, padding=1, dropout_prob=0.1, relu=decoder_relu)
            self.bottleneck4_2 = Bottleneck(self.nch, padding=1, dropout_prob=0.1, relu=decoder_relu)            
            
            # Stage 5 - Decoder
            
            '''
            self.bottleneck5_0 = Bottleneck(in_channels=64, out_channels=16, dropout_prob=0.1)
            self.bottleneck5_1 = Bottleneck(in_channels=16 + 64, out_channels=16, dropout_prob=0.1)

            # Final Convolution
            self.final_conv = nn.Conv2d(in_channels=16, out_channels=self.n_classes, kernel_size=1)
            '''
            self.upsample5_0 = UpsamplingBottleneck(
                self.nch, self.nch_val, dropout_prob=0.1, relu=decoder_relu)
            self.bottleneck5_1 = Bottleneck(
                self.nch_val, padding=1, dropout_prob=0.1, relu=decoder_relu)
            self.final_conv = nn.ConvTranspose2d(
                self.nch_val,
                self.n_classes,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False)            
            


    def forward(self, x):
        # Initial Block
        input_size = x.size()
        x = self.initial_block(x)
        
    
        # Stage 1 - Encoder
        stage1_input_size = x.size()
        #print('initial shape : {}'.format(x.shape))
        #initial shape : torch.Size([8, 16, 192, 608])
        
        
        
        
        x, max_indices1_0 = self.downsample1_0(x)
        #print('down shape : {}'.format(x.shape))
        #down shape : torch.Size([8, 64, 96, 304])
        
        x = self.bottleneck1_1(x)
        #print('bottle-1 shape : {}'.format(x.shape))
        #bottle-1 shape : torch.Size([8, 64, 96, 304])
        x = self.bottleneck1_2(x)
        #print('bottle-2 shape : {}'.format(x.shape))
        #bottle-2 shape : torch.Size([8, 64, 96, 304])
        x = self.bottleneck1_3(x)
        x = self.bottleneck1_4(x)

        # Stage 2 - Encoder
        # Stage 2 - Encoder
        stage2_input_size = x.size()
        x, max_indices2_0 = self.downsample2_0(x)
        x = self.bottleneck2_1(x)
        x = self.bottleneck2_2(x)
        x = self.asybottleneck2_3(x)
        x = self.bottleneck2_4(x)
        x = self.bottleneck2_5(x)
        x = self.bottleneck2_6(x)
        x = self.asybottleneck2_7(x)
        x = self.bottleneck2_8(x)

        # Stage 3 - Encoder
        x = self.bottleneck3_1(x)
        x = self.bottleneck3_2(x)
        x = self.asybottleneck3_3(x)
        x = self.bottleneck3_4(x)
        x = self.bottleneck3_5(x)
        x = self.bottleneck3_6(x)
        x = self.asybottleneck3_7(x)
        x = self.bottleneck3_8(x)
        
        
        # Stage 4 - Decoder
        x = self.upsample4_0(x, max_indices2_0, output_size=stage2_input_size)
        x = self.bottleneck4_1(x)
        x = self.bottleneck4_2(x)

        # Stage 5 - Decoder
        x = self.upsample5_0(x, max_indices1_0, output_size=stage1_input_size)
        x = self.bottleneck5_1(x)
        x = self.final_conv(x, output_size = input_size)

        return x
    
'''                                        
class EnsembleNet(nn.Module):
    def __init__(self, model_name, n_ch, n_cls):
        super(EnsembleNet, self).__init__()
        
        self.model_name = model_name
        self.n_channels = n_ch
        self.n_classes = n_cls
        

        if 'ensemble' in self.model_name:

            self.nch = 64
            
            self.encoder1 = Encoder(self.n_channels, self.nch)
            self.encoder2 = Encoder(self.nch, self.nch * 2)
            self.encoder3 = Encoder(self.nch * 2, self.nch * 4)
            self.encoder4 = Encoder(self.nch * 4, self.nch * 8)

            # Attention Blocks 추가
            #self.attention1 = AttentionBlock(self.nch * 8, self.nch * 8, self.nch * 4)  # encoder4와 decoder4 사이
            #self.attention2 = AttentionBlock(self.nch * 4, self.nch * 4, self.nch * 2)   # encoder3와 decoder3 사이
            #self.attention3 = AttentionBlock(self.nch * 2, self.nch * 2, self.nch)     # encoder2와 decoder2 사이
            #self.attention4 = AttentionBlock(self.nch, self.nch, self.nch)     # encoder1와 decoder1 사이

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

            # Attention Mechanism
            #x = self.attention1(x, x)  # encoder4의 출력에 Attention 적용
            x = self.decoder4(x, indices4)

            #x = self.attention2(x, x)  # encoder3의 출력에 Attention 적용
            x = self.decoder3(x, indices3)

            #x = self.attention3(x, x)  # encoder2의 출력에 Attention 적용
            x = self.decoder2(x, indices2)

            #x = self.attention4(x, x)  # encoder1의 출력에 Attention 적용
            x = self.decoder1(x, indices1)
            
            # 최종 출력
            out = self.final_conv(x)
            
            return out     
'''


