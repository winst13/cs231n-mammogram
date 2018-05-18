import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as T
import numpy as np
import os

class MammogramDataset(Dataset):
    def __init__(self, root_dir, dataset, tranform = None):
        """ Params:
            |root_dir|: The dir under which we have the 
                training and test splits, usually [reporoot]/data/
            |dataset|: one of "train", "test"
            |transform|: A function which is applied to each datapoint before retrieval, if provided.
        """
        self.root_dir = root_dir
        self.dataset = dataset
        self.tranform = tranform
        self.datapoints = []
        with open(os.path.join(root_dir, dataset+".txt"), 'r') as file:
            for line in file:
                self.datapoints.append(tuple(line.split))

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):
        label, image_name = self.datapoint[idx]
        image_path = os.path.join(self.root_dir, self.dataset, label, image_name)
        with open(image_path, 'r') as image_file:
            image = np.load(image_file)
        sample = {'image': image, 'label': label}
        
        if self.transform:
            sample = self.transform(sample)

        return sample

#Examples of how to use:
#train_dataset = MammogramDataset("data", "train")
#test_dataset = MammogramDataset("data", "test")


