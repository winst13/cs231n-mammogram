import os
from os.path import join, isdir, exists


def ensure_dir_created(*args):
    for path in args:
        # os.makedirs: Recursively make intermediate dirs if necessary
        if not isdir(path): os.makedirs(path)    
        


def get_the_only_directory_under(dirpath):
    """ Get the first (and the only) directory under a path.
    """
    dirs = [name for name in os.listdir(dirpath) if isdir(join(dirpath, name))]
    if len(dirs) != 1:
        raise ValueError("In 'get_the_only_directory_under' call, "
            "found more than 1 directory under: %s" % dirpath)
    return dirs[0]
