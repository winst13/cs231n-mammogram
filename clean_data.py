import os
from os.path import join, isdir, exists
import pydicom as dicom
import numpy as np
from scipy.misc import imsave, imresize

#from matplotlib import pyplot as plt

from util.image import normalize_between


# Constants for this script
data_dir = "data"
masstest_datapath = join(data_dir, "mass-test")
# Inside this dir (above line), there should reside multiple folders like 
# "Mass-Test_P_00016_LEFT_CC / 10-04-2016-DDSM-30104 / 1-full mammogram images-14172 / 000000.dcm"

masstest_outpath = join(data_dir, "mass-test-out")
# Output dir


def get_the_only_directory_under(dirpath):
    """ Get the first (and the only) directory under a path.
    """
    dirs = [name for name in os.listdir(dirpath) if isdir(join(dirpath, name))]
    if len(dirs) != 1:
        raise ValueError("In 'get_the_only_directory_under' call, "
            "found more than 1 directory under: %s" % dirpath)
    return dirs[0]


def dig_out_dcm_image(dirpath, nb_layers=2):
    """
    Remove the useless directory layers, get a filepath to the
    dicom image file.
    """

    # There should be 3 layers of useless dirs
    metadata = []
    onion = dirpath

    for _ in range(nb_layers):
        onion_peel = get_the_only_directory_under(onion)
        onion = join(onion, onion_peel)
        metadata.append(onion_peel)

    imgname = "000000.dcm" # This seems to be the name for all of them
    filepath = join(onion, imgname)
    
    return filepath, metadata


def dcm_as_numpy_array(filepath):
    ds = dicom.dcmread(filepath)
    imgarray = ds.pixel_array
    #print("imgarray:\n", imgarray, type(imgarray), imgarray.shape)
    shape = imgarray.shape
    return imgarray, shape


def preprocess(imgarray):
    imgarray = normalize_between(imgarray, 0, 1)
    imgarray = imgarray.astype(np.float32)
    
    size = (1024, 1024)
    imgarray = imresize(imgarray, size, interp='bicubic')

    return imgarray

def preprocess_for_jpg(imgarray, resize=None):
    imgarray = normalize_between(imgarray, 0, 255)
    if resize is not None:
        size = (resize, resize)
        imgarray = imresize(imgarray, size, interp='bicubic')
    return imgarray


def clean_data(dirpath, outpath):

    shape_datapts = []
    names_so_far = []

    for name in os.listdir(dirpath):
        print("We are doing the image for:", name)
        assert name not in names_so_far
        names_so_far.append(name)

        imgpath = join(dirpath, name)
        # We only want directories, which have the dicom images under them
        if not isdir(imgpath):
            print("Not a dir, continuing")
            continue
        
        # Get the dicom and turn it into ndarray
        filepath, path_metadata = dig_out_dcm_image(imgpath)
        imgarray, shape = dcm_as_numpy_array(filepath)
        shape_datapts.append(shape)

        # Store each sample!
        imgarray_for_npy = preprocess(imgarray)
        imgarray_for_jpg = preprocess_for_jpg(imgarray, resize=1024)
        #imgarray_for_jpg_noresize = preprocess_for_jpg(imgarray)
        outpath_npy = join(outpath, name + ".npy")
        outpath_img = join(outpath, name + ".jpg")
        #outpath_imgnr = join(outpath, name + "nr.jpg")
        
        np.save(outpath_npy, imgarray_for_npy)
        # imsave expects from 0-255
        imsave(outpath_img, imgarray_for_jpg)
        #imsave(outpath_imgnr, imgarray_for_jpg_noresize)

    # Plotting the shapes of our data samples in a scatter plot
    #plt.plot(*zip(*shape_datapts), 'bo')
    #plt.show()


if __name__ == "__main__":
    clean_data(masstest_datapath, masstest_outpath)