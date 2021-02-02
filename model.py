import torch
import torch.nn as nn
import functools
import itertools

import numpy as np

############################################################################
### ResNet
### Code from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/ 
### with extension of various padding types
############################################################################

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class Resnet_2D(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, 
                 use_dropout=False, n_blocks=6, padding_type='reflect', 
                 n_downsampling = 2):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
            n_downsampling -- number of downsamplings
        """
        assert(n_blocks >= 0)
        super(Resnet_2D, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
            
        print("\n------Initiating ResNet------\n")

        model = []
        p = 0
        if padding_type == 'reflect':
            model += [nn.ReflectionPad2d(3)]
        elif padding_type == 'replicate':
            model += [nn.ReplicationPad2d(3)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
            
        model += [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=p, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
            
        p = 0
        if padding_type == 'reflect':
            model += [nn.ReflectionPad2d(3)]
            model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        elif padding_type == 'replicate':
            model += [nn.ReplicationPad2d(3)]
            model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        elif padding_type == 'zero':
            model += [nn.ConvTranspose2d(ngf, output_nc, kernel_size=7, padding=1)]
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
            
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)
        self.input_nc = input_nc
        self.output_nc = output_nc

    def forward(self, input):
        """Standard forward"""
        return self.model(input)
############################################################################

############################################################################
### Integration scheme
############################################################################
class Integrate(nn.Module):
    def __init__(self, dyn, order=1, n_steps=1, dt=1):
        super(Integrate, self).__init__()
        self.dyn = dyn
        self.order = order
        self.n_steps = n_steps
        self.dt = dt*1.0/self.n_steps
        
    def model_broadcasting(self, x):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        x_out = torch.zeros(x.size(), device=device)
        x_out_model = self.dyn(x)
        
        nb_ch_in = x.size()[1]
        nb_ch_out = x_out_model.size()[1]
        
        channel = np.sign(nb_ch_in-nb_ch_out)
        
        if channel>0:
            x_out[:,1:(nb_ch_out+1),] += x_out_model[:,:,:x.size()[-2],:x.size()[-1]]
        elif channel==0:
            x_out[:,:,] += x_out_model[:,:,:input.size()[-2],:input.size()[-1]]
        else : 
            raise NotImplementedError('Output channels > input channels is not implemented')
            
        return x_out

    def forward(self, x):
        for t in range(self.n_steps):
            if self.order == 1:
                x = x + self.dt*self.model_broadcasting(x)
            elif self.order == 2:
                x = x + self.dt*self.model_broadcasting(x+0.5*self.dt*self.model_broadcasting(x))
            elif self.order == 4:
                k1 = self.model_broadcasting(x)
                k2 = self.model_broadcasting(x+0.5*self.dt*k1)
                k3 = self.model_broadcasting(x+0.5*self.dt*k2)
                k4 = self.model_broadcasting(x+self.dt*k3)
                x = x + self.dt*(k1 + 2*k2 + 2*k3 + k4)/6.0
            else:
                raise NotImplementedError
            
        return x
############################################################################