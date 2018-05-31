import os

datadir = "data"

def write_data_list(dataset):
    with open(os.path.join(datadir, dataset+".txt"), 'w') as file:
        dir = os.path.join(datadir, dataset)
        #negative examples
        negative_dir = os.path.join(dir, "0")
        for filename in os.listdir(negative_dir):
            if filename.endswith(".npy"):
                file.write("0 "+filename+"\n")
        positive_dir = os.path.join(dir, "1")
        for filename in os.listdir(positive_dir):
            if filename.endswith(".npy"):
                file.write("1 "+filename+"\n")

datasets = ["test", "train"]

for i in datasets:
    write_data_list(i)
    
# Note:  ran /usr/local/bin/gshuf -o test.txt < test.txt and similar for train as well
