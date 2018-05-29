import os
import numpy as np

def get_stats(folder_paths):
    num_pix = 0
    tot = 0
    tot2 = 0
    for dirpath in folder_paths:
        for name in os.listdir(dirpath):
            if name.endswith(".npy"):
                path = os.path.join(dirpath, name)
                img = np.load(path)
                num_pix += np.prod(img.shape)
                tot += np.sum(img)
                tot2 += np.sum(img**2)
    mean = tot/num_pix
    var = tot2/num_pix - mean**2
    return mean, np.sqrt(var)
                
                
if __name__ == "__main__":
    mean, std = get_stats(["./data/train/0", "./data/train/1"])
    print ("mean = ", mean, ", std = ", std)