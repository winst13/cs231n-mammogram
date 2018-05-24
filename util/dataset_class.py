import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as T
import numpy as np
import os

class MammogramDataset(Dataset):
    def __init__(self, root_dir, dataset, transform = None):
        self.root_dir = root_dir
        self.dataset = dataset
        self.transform = transform
        self.datapoints = []
        with open(os.path.join(root_dir, dataset+".txt"), 'r') as file:
            for line in file:
                self.datapoints.append(tuple(line.split()))

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):
        label, image_name = self.datapoints[idx]
        image_path = os.path.join(self.root_dir, self.dataset, label, image_name)
        with open(image_path, 'r') as image_file:
            image = np.load(image_file)
        sample = {'image': image, 'label': int(label)}
        
        if self.transform:
            sample = self.transform(sample)

        return sample

#Examples of how to use:
#train_dataset = MammogramDataset("data", "train")
#test_dataset = MammogramDataset("data", "test")


