import torch as torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import densenet
from torchvision.models.densenet import _DenseBlock, _Transition
import torchsummary

#import code
from copy import deepcopy
from collections import OrderedDict

# Project imports
from util.util import print # print with a header for this project


# Leveraged Torchvision implementation of DenseNet at:
#   https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

_use_pretrained = True
name_to_model_fns = {
      "densenet121" : lambda: densenet.densenet121(pretrained=_use_pretrained)
    , "densenet169" : lambda: densenet.densenet169(pretrained=_use_pretrained)
    , "densenet201" : lambda: densenet.densenet201(pretrained=_use_pretrained) # We're just going to use this one
    , "densenet161" : lambda: densenet.densenet161(pretrained=_use_pretrained)
}

def get_densenet(name):
    return name_to_model_fns[name]() # Call the function

"""
old_model.modules is more accurate of all indiv. layers
    old_model.features decomposes structure more, but doesn't include final FC classifier
    (instead under old_model.classifier)

DenseNet201's model.features [nn.Sequential]
    0 = conv2d
    1 = batchnorm
    2 = relu
    3 = maxpool
    4 = _DenseBlock (6 DenseLayer's)
    5 = _Transition
    6 = _DenseBlock (12 DenseLayer's)
    7 = _Transition
    8 = _DenseBlock (48 DL's)
    9 = _Transition
    10 = _DenseBlock (32 DL's)
    11 = BatchNorm
Final linear layer:
    self.classifier = nn.Linear(
        num_features=[code computes this],
        num_classes=1000)
"""


def preprocess(x):
    """ Sample preprocessing code and statistics examples are at:
        https://github.com/titu1994/DenseNet/blob/master/densenet.py#L37
        
        Our data from the .npy mammogram image tensors are already normed to [0,1].
        _dataset_mean = 0.217989
        _dataset_std  = 0.257150
        
    """    
    return (x - 0.217989) / 0.257150



def transform_filters_to_grayscale(m):
    """ Turn the filter tensors for Conv2d expecting RGB to grayscale.
            Use average method; no need to scale with 0.3R + 0.59G + 0.11B
            because not for human eyes.

        param |m|: pytorch Conv2d module/layer
    """
    assert m.weight.size()[1] == m.in_channels == 3
    has_bias = m._parameters['bias'] is not None
    # Should be no bias. Good to have for code reuse in the future though
    if has_bias:
        raise Exception("WARNING: The input module param to |transform_filters_to_grayscale| has a bias")
    
    new_conv = nn.Conv2d(1, m.out_channels, m.kernel_size, 
                         stride=m.stride, padding=m.padding, 
                         dilation=m.dilation, groups=m.groups, bias=has_bias)

    weight_tensor = m.weight.data.mean(dim=1, keepdim=True) # (out, 1, {2}*kernel_size)
    new_conv.weight.data = weight_tensor
    if has_bias:
        new_conv.bias.data = m.bias.data.clone()

    return new_conv


def freeze_parameters(module):
    for param in module.parameters():
        # Even for classes like _DenseBlock, all Parameter objs
        #  from submodules are registered recursively
        param.requires_grad = False



def summary(model, input_size=(1, 1024, 1024)):
    """ Do a Keras-like summary of the pytorch model.
        param |model|: nn.Module subclass or -.Sequential model
    """
    # Assume method will be used on MammogramDenseNet by default
    torchsummary.summary(input_size)


def get_pretrained_layers(model_name='densenet201'):

    # Use Densenet-201 by default

    print("Retrieving the pretrained DenseNet model:", model_name)
    old_model = get_densenet(model_name)
    
    #new_model = densenet.DenseNet(
    #    growth_rate=32, block_config=(6, 12, 6),
    #    num_init_features=64, drop_rate=0., num_classes=2)
    # TODO(ojwang): Increase drop rate when model is confirmed working and time to tune

    layers = []

    # The first Conv2d layer
    print("Copying features[0]: %s..." % old_model.features[0])
    first_conv = transform_filters_to_grayscale(old_model.features[0])
    layers.append(('conv0', first_conv))
    # Don't freeze this first layer: We will keep training it, because the colors
    #  are not perfect for grayscale images, and our preprocessing norm statistics
    #  are different from original DenseNet trained on ImageNet

    # The initial BatchNorm
    print("Copying features[1]: %s..." % old_model.features[1])
    first_bn = deepcopy(old_model.features[1])
    freeze_parameters(first_bn)
    layers.append(('batchnorm0', first_bn))

    # the classic ReLU
    print("Creating new ReLU for features[2]: %s..." % old_model.features[2])
    first_relu = nn.ReLU(inplace=True)
    layers.append(('relu0', first_relu))

    # The initial BatchNorm
    print("Copying features[3]: %s..." % old_model.features[3])
    first_maxpool = deepcopy(old_model.features[3])
    layers.append(('maxpool0', first_maxpool))

    # The first dense block: 6 dense layers
    #   (each is 1x1 conv -> 3x3 conv, w/ appropriate relu and batchnorm)
    print("Copying features[4]: %s..." % type(old_model.features[4])) # print only type: DB has too much text
    denseblock = deepcopy(old_model.features[4])
    freeze_parameters(denseblock)
    layers.append(('denseblock0', denseblock))

    layers = OrderedDict(layers)

    return layers


class MammogramDenseNet(nn.Module):
    """ Description
    """

    def __init__(self, growth_rate=32, block_config=(12,6),
                 bn_size=4, drop_rate=0, pretrained_encoder=True):

        super(MammogramDenseNet, self).__init__()
        
        self.num_classes = 2

        if pretrained_encoder:
            pretrained_layers = get_pretrained_layers() # Densenet-201 default
            self.features = nn.Sequential(pretrained_layers)

        # Add the rest of the architecture (Dense blocks, transition layers)
        # self.features.add_module(...)


        """
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        """


        # Put the classifier here separately (will apply it manually in forward(x))
        self.classifier = None # nn.Linear(...)


        # The official init loop from the PyTorch repo for densenet.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
        """


    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True) # Last Relu, bc most recent was a conv

        # Note: The .view() below is probably why the classifier isn't included as a module
        # out = F.avg_pool2d(out, kernel_size=7, stride=1)
        # out = out.view(features.size(0), -1)

        # Classifier in __init__ requires knowing # of neurons at this flattened stage
        out = self.classifier(out)




