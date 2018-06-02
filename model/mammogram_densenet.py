import torch as torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import densenet
from torchvision.models.densenet import _DenseBlock, _Transition

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
    torchsummary.summary(model, input_size)


def get_pretrained_layers(model_name='densenet201', include_denseblock=True):

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
    # Don't freeze layers: We will keep training it, because the domain isn't the same as ImageNet
    
    # The initial BatchNorm
    print("Copying features[1]: %s..." % old_model.features[1])
    first_bn = deepcopy(old_model.features[1])
    #freeze_parameters(first_bn)
    layers.append(('batchnorm0', first_bn))

    # the classic ReLU
    print("Creating new ReLU for features[2]: %s..." % old_model.features[2])
    first_relu = nn.ReLU(inplace=True)
    layers.append(('relu0', first_relu))

    # The initial MaxPool
    print("Copying features[3]: %s..." % old_model.features[3])
    first_maxpool = deepcopy(old_model.features[3])
    layers.append(('maxpool0', first_maxpool))

    if include_denseblock:
        # The first dense block: 6 dense layers
        #   (each is 1x1 conv -> 3x3 conv, w/ appropriate relu and batchnorm)
        print("Copying features[4]: %s..." % type(old_model.features[4])) # print only type: DB has too much text
        denseblock = deepcopy(old_model.features[4])
        #freeze_parameters(denseblock)
        layers.append(('denseblock0', denseblock))

    layers = OrderedDict(layers)

    return layers

class Swish(nn.Module):
    """ Swish activation by Google that works better for 40-50+ layer networks.
    """
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(data=torch.ones(1))
        self.beta.requires_grad = True

    def forward(self, x):
        #beta = self.beta.expand_as(x)
        return x * F.sigmoid(self.beta * x)


class MammogramDenseNet(nn.Module):
    """ Description
    """

    def __init__(self, growth_rate=32, block_config=(6,12,18,12),
                 bn_size=4, drop_rate=0., pretrained_encoder=2, debug=False):
        """
        bn_size = bottleneck size, the factor by which the first conv in a _DenseLayer
            is larger than the second conv.
        :pretrained_encoder: int in [0,1,2] designating level of pretrained layers to use.
            0 is none, 1 is just first convolutions, 2 is the first dense block.
        """

        super(MammogramDenseNet, self).__init__()
        
        self.debug = debug
        self.pretrained = pretrained_encoder
        self.nb_dense_blocks = len(block_config)
        num_classes = 2  # Benign (0) or Malignant (1)

        if self.pretrained > 0:
            include_denseblock = self.pretrained == 2
            pretrained_layers = get_pretrained_layers(include_denseblock=include_denseblock)
            self.features = nn.Sequential(pretrained_layers)

            # Display shapes for debugging
            if debug:
                print("pretrained_encoder = %d" % pretrained_encoder)
                print("Output shape after the pretrained modules (batch, channels, H, W):")
                # summary() printout takes a lot of time, but more comprehensive info
                #summary(self.features)

                test_input = torch.rand(1,1,1024,1024)
                test_output = self.features(test_input)
                print(test_output.size())
                del test_input
                del test_output
                # test_output: (-1, 256, 256, 256)
        else:
            print("No pretrained layers.")
            self.features = nn.Sequential() # Empty model if no pretrained encoder

        # A counter to track what input shape our final nn.Linear layer should expect
        #  Just num_channels is fine, because global avg pool at end
        num_features = 256 if self.pretrained == 2 else (64 if self.pretrained == 1 else 1)

        # Add the rest of the architecture (Dense blocks, transition layers)
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)

            # Initialize the weights of block
            for m in block.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                    # Conv layers have no bias when in conjunction with Batchnorm
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            self.features.add_module('denseblock%d' % (i + 1), block)
            
            num_features = num_features + num_layers * growth_rate
            if debug: print("num features after denseblock %d:" % (i + 1), num_features)

            # Add a transition layer if not the last dense block:
            #  Norm, 1x1 Conv, (activation), AvgPool
            if i != self.nb_dense_blocks - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features)
                self.features.add_module('transition%d' % (i + 1), trans)
                if debug: print("num features after transition %d:" % (i + 1), num_features)

        if debug: print("final num features:", num_features)

        # Put the classifier here separately 
        #  will apply it manually in forward(x), after global avg pool and reshape
        self.classifier = nn.Linear(num_features, num_classes)
        nn.init.constant_(self.classifier.bias, 0)


    def preprocess(self, x):
        """ Sample preprocessing code and statistics examples are at:
            https://github.com/titu1994/DenseNet/blob/master/densenet.py#L37
            
            Our data from the .npy mammogram image tensors are already normed to [0,1].
            _dataset_mean = 0.217989
            _dataset_std  = 0.257150
            
        """
        return (x - 0.217989) / 0.257150


    def forward(self, x):
        x = self.preprocess(x)

        features = self.features(x)
        if self.debug: print("After all convolutions:", features.size())

        #out = F.relu(features, inplace=True) # Last Relu, bc most recent was a conv
        final_swish = Swish().cuda()
        out = final_swish.forward(features)
        print("out.size() =", out.size(), "| Number of zeros after final relu:", (out == 0).sum())

        # Global average pooling
        resolution = out.size(2)
        out = F.avg_pool2d(out, kernel_size=(resolution, resolution), stride=1)
        if self.debug: print("After avg pool:", out.size())

        out = out.view(features.size(0), -1)
        
        # Classifier created in __init__
        out = self.classifier(out)
        if self.debug: print(out.size())
        return out


