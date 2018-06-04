import torch as torch
from torch import nn
from torch.nn import functional as F
from util.custom_layers import Flatten

class BaselineModel(nn.Module):
    def __init__(self, growth_rate=32, block_config=(12,6),
                 bn_size=4, drop_rate=0, pretrained_encoder=True, debug=False):

        super(BaselineModel, self).__init__()
        
        self.num_classes = 2  # Benign (0) or Malignant (1)
        
        def init_weights(model):
            if type(model) == nn.Linear:
                torch.nn.init.xavier_normal_(model.weight)
                model.bias.data.fill_(0)
            if type(model) == nn.Conv2d:
                torch.nn.init.kaiming_normal_(model.weight)
                model.bias.data.fill_(0) 
        
        self.features = nn.Sequential(
            nn.Dropout2d(drop_rate),
            nn.Conv2d(1, 3, 3, padding=1),
            nn.Dropout2d(drop_rate),
            nn.MaxPool2d(2),
            nn.Conv2d(3, 3, 3, padding=1),
            nn.Dropout2d(drop_rate),
            nn.MaxPool2d(2),
            nn.Conv2d(3, 3, 3, padding=1),
            nn.Dropout2d(drop_rate),
            nn.MaxPool2d(2),
            Flatten(),
            nn.Linear(3*128*128, 1),
            nn.Sigmoid()
        )
        self.features.apply(init_weights)  
            
    def forward(self, x):
        features = self.features(x)
        return features