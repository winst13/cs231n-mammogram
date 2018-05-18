import torch
import numpy as np
from util.dataset_class import MammogramDataset

train_data = MammogramDataset("data", "train")
test_data = MammogramDataset("data", "test")



