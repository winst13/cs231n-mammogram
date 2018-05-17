
import csv
import numpy as np
import os
from os.path import join
from shutil import copyfile

from util.path import ensure_dir_created

data_dir = "data"
label_dir = "labels_raw_csv"
splits_csv_names = [
    "Mass-Test"
    , "Mass-Train"
    , "Calc-Test"
    , "Calc-Train"
]

def get_label_fn_parameters(prefix):
    def csvpath(p): return join(label_dir, prefix + '.csv')
    #def outpath(p): return join(data_dir, prefix + "-out")
    #def outpath_for_jpg(p): return join(data_dir, prefix + "-out-jpg")
    #return prefix, datapath(prefix), outpath(prefix), outpath_for_jpg(prefix)

    return prefix, csvpath(prefix)

def get_csvreader_from_filepath(path):
    csvfile = open(path, newline='') # We do not scope the open() call, want to REMAIN open!
    reader = csv.DictReader(csvfile, fieldnames=None) # Infer fieldnames from first row
    return reader

def get_label(row):
    def is_malignant(string): # Based on labels in csv. There are multiple classes of labels.
        return "MALIGNANT" in string
    def is_benign(string): # Based on labels in csv
        return "BENIGN" in string
    label_field = row["pathology"]
    if is_malignant(label_field) and not is_benign(label_field):
        return 1
    elif is_benign(label_field) and not is_malignant(label_field):
        return 0
    else:
        raise ValueError("Unrecognized field, cannot tell if benign (0) or mal. (1)")

def construct_sample_name(row):
    """ param |row|: An OrderedDict representing one row in the csv.
        string tuples of (fieldname, content).
    """
    patient_id = row["patient_id"]
    assert len(patient_id) == 7 and patient_id[:2] == 'P_'
    LR = row["left or right breast"]
    assert LR in ("LEFT", "RIGHT")
    view = row["image view"]
    assert view in ("MLO", "CC")
    # Note, dangerous: Abusing python scope w/ _curr_prefix
    sample_name = '_'.join([_curr_prefix, patient_id, LR, view])
    # e.g. "Mass-Test_P_00296_LEFT_MLO"

    print("Found row in csv with sample name:", sample_name)
    return sample_name


def build_sample_to_class_map(reader):
    sample_to_class = {}
    for row in reader:
        sample_name = construct_sample_name(row)
        class_label = get_label(row)

        # Collect all labels for the tumor(s) in that mammogram
        if sample_name in sample_to_class:
            sample_to_class[sample_name].append(class_label)
        else:
            sample_to_class[sample_name] = [class_label]

    # Deal with multiple labels for one mammogram:
    #   All agree on label = keep, otherwise discard sample
    for sample, classes in sample_to_class.items():
        if len(classes) == 1:
            continue

        @

    return sample_to_class


def setup_directories(prefix):
    dirname = prefix.lower() + "-labeled"
    class0_dir = join(data_dir, dirname, '0')
    class1_dir = join(data_dir, dirname, '1')
    ensure_dir_created(class0_dir, class1_dir)
    return class0_dir, class1_dir



def assign_labels(split_prefix, csvpath):
    print("Reading csv from:", csvpath)
    reader = get_csvreader_from_filepath(csvpath)
    labeled_samples = build_sample_to_class_map(reader)
    print("Built mapping to labels for %d samples in csv." % len(labeled_samples))

    class0_dir, class1_dir = setup_directories(split_prefix)
    split_output_dir = join(data_dir, split_prefix.lower() + "-out")
    unsorted_split_files = [name for name in os.listdir(split_output_dir) if name.endswith('.npy')]
    print("Found %d samples under %s." % (len(unsorted_split_files), split_output_dir))

    assert len(unsorted_split_files) == len(labeled_samples) # Sanity check: label for every sample?

    for fname in unsorted_split_files:
        sample_id = fname[:-4]   # Remove the .npy extension
        klass = labeled_samples[sample_id]
        dstpath = join(class0_dir if klass == 0 else class1_dir, fname)
        srcpath = join(split_output_dir, fname)

        #os.rename(srcpath, dstpath) # Too dangerous, just copy it first
        copyfile(srcpath, dstpath)
    


################################

if __name__ == "__main__":
    for _curr_prefix in splits_csv_names:
        parameters = get_label_fn_parameters(_curr_prefix)
        assign_labels(*parameters)

        print("We are still debugging! Breaking early")
        break