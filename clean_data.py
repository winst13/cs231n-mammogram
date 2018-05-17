import os
from os.path import join, isdir
import pydicom as dicom
import numpy as np
from scipy.misc import imsave, imresize

from matplotlib import pyplot as plt

from util.image import normalize_between
from util.path import get_the_only_directory_under


# Constants for this script
data_dir = "data"
_dicom_imgname = "000000.dcm" # This seems to be the name for all of them

data_splits = [
    "mass-test"
    , "mass-train"
    , "calc-test"
    , "calc-train"
]

def get_path_parameters(prefix):
    # Inside this datapath dir, there should reside multiple folders like:
    # "Mass-Test_P_00016_LEFT_CC / 10-04-2016-DDSM-30104 / 1-full mammogram images-14172 / 000000.dcm"
    def datapath(p): return join(data_dir, prefix)
    def outpath(p): return join(data_dir, prefix + "-out")
    def outpath_for_jpg(p): return join(data_dir, prefix + "-out-jpg")
    return prefix, datapath(prefix), outpath(prefix), outpath_for_jpg(prefix)


#####################



def dig_out_dcm_image(dirpath, nb_layers=2):
    """ Remove the useless directory layers, get a filepath to the
        dicom image file.
    """

    # There should be 3 layers of useless dirs
    #metadata = []
    onion = dirpath

    for _ in range(nb_layers):
        onion_peel = get_the_only_directory_under(onion)
        onion = join(onion, onion_peel)
        #metadata.append(onion_peel)
    filepath = join(onion, _dicom_imgname)
    return filepath #, metadata


def dcm_as_numpy_array(filepath):
    ds = dicom.dcmread(filepath)
    imgarray = ds.pixel_array
    #print("imgarray:\n", imgarray, type(imgarray), imgarray.shape)
    shape = imgarray.shape
    return imgarray, shape


def preprocess(imgarray):
    size = (1024, 1024)
    imgarray = imresize(imgarray, size, interp='bicubic', mode='F')
    imgarray = normalize_between(imgarray, 0, 1)
    imgarray = imgarray.astype(np.float32)
    
    return imgarray

def preprocess_for_jpg(imgarray, resize=None):
    #imgarray = normalize_between(imgarray, 0, 255)
    if resize is not None:
        size = (resize, resize)
        imgarray = imresize(imgarray, size, interp='bicubic')
        # Resizing automatically norms to 0-255 and np.uint8 type, with default mode=None->'L'
        
    return imgarray

def save_plot_of_datapoint_shapes(shape_datapts, plot_save_path):
    plt.plot(*zip(*shape_datapts), 'bo')
    plt.xlabel("Width (pixels)")
    plt.ylabel("Height (pixels)")
    plt.savefig(plot_save_path)

def clean_data(datasplit_prefix, dirpath, outpath, outpath_for_images, collect_shapes_only=False):

    shape_datapts = []

    for nb, name in enumerate(os.listdir(dirpath)):
        print("We are doing the image for:", name)

        imgpath = join(dirpath, name)
        # We only want directories, which have the dicom images under them
        if not isdir(imgpath):
            print("Not a dir, continuing")
            continue
        
        #print("flag 1")
        # Get the dicom and turn it into ndarray
        #filepath, path_metadata = dig_out_dcm_image(imgpath)
        filepath = dig_out_dcm_image(imgpath)  # THis line expects no metadata return value
        imgarray, shape = dcm_as_numpy_array(filepath)
        shape_datapts.append(shape)
        if collect_shapes_only:  # Use case: error stopped us from plotting, go through again only with plotting.
            continue

        #print("flag 2")
        # Store each sample Numpy!
        imgarray_for_npy = preprocess(imgarray)
        savepath_npy = join(outpath, name + ".npy")
        np.save(savepath_npy, imgarray_for_npy)

        #print("flag 3")
        # Store each sample jpg (for details later)! One every 10 should be fine
        if nb % 10 == 0:
            imgarray_for_jpg = preprocess_for_jpg(imgarray, resize=1024)
            savepath_img = join(outpath_for_images, name + ".jpg")
            imsave(savepath_img, imgarray_for_jpg)     # imsave expects from 0-255

        #imgarray_for_jpg_noresize = preprocess_for_jpg(imgarray)
        #outpath_imgnr = join(outpath, name + "nr.jpg")
        
        #imsave(outpath_img+"_checkNumpy.jpg", normalize_between(imgarray_for_npy, 0, 255).astype(np.uint8))
        #imsave(outpath_imgnr, imgarray_for_jpg_noresize)

    # Plotting the shapes of our data samples in a scatter plot, for details later
    plot_save_path = join("assets", datasplit_prefix + "datapoints_heightwidth.png")
    save_plot_of_datapoint_shapes(shape_datapts, plot_save_path)


if __name__ == "__main__":
    for data_split_prefix in data_splits:
        #clean_data(*get_path_parameters(data_split_prefix), collect_shapes_only=True)
        clean_data(*get_path_parameters(data_split_prefix))
        
        #break



