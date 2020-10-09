import torch.nn as nn
from torch.nn import init
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
import torch.nn.functional as functional
from torch.autograd import Variable

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, padding_mode='reflect')
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, padding_mode='reflect')
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1, padding_mode='reflect')
        self.conv5 = nn.Conv2d(256, 128, kernel_size=3, padding=1, padding_mode='reflect')
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1, padding_mode='reflect')
        self.conv7 = nn.Conv2d(128, 64, kernel_size=3, padding=1, padding_mode='reflect')
        self.conv8 = nn.Conv2d(64, 64, kernel_size=3, padding=1, padding_mode='reflect')
        self.conv9 = nn.Conv2d(64, 3, kernel_size=3, padding=1, padding_mode='reflect')

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = nn.Upsample(size=(64, 128), mode='nearest')(x)

        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = nn.Upsample(size=(128, 256), mode='nearest')(x)

        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = nn.Upsample(size=(256, 512), mode='nearest')(x)

        x = self.relu(self.conv8(x))
        x = self.conv9(x)

        return x


class VGG19features(nn.Module):
    def __init__(self):
        super(VGG19features, self).__init__()
        self.conv0 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv7 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv10 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv12 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv14 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv16 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv19 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv21 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv23 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv25 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv28 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv30 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv32 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv34 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

    def forward(self, x):
        x = self.relu(self.conv0(x))
        x = self.maxpool(self.relu(self.conv2(x)))

        x = self.relu(self.conv5(x))
        x = self.maxpool(self.relu(self.conv7(x)))

        x = self.relu(self.conv10(x))
        x = self.relu(self.conv12(x))
        x = self.relu(self.conv14(x))
        x = self.maxpool(self.relu(self.conv16(x)))

        x = self.relu(self.conv19(x))
        x = self.relu(self.conv21(x))
        x = self.relu(self.conv23(x))
        x = self.relu(self.conv25(x))
        x1 = self.maxpool(x)

        x1 = self.relu(self.conv28(x1))
        x1 = self.relu(self.conv30(x1))
        x1 = self.relu(self.conv32(x1))
        x1 = self.maxpool(self.relu(self.conv34(x1)))

        return x

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    # print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>
    return net


