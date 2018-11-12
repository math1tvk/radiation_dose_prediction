import numpy as np
import os
import h5py


# filename to save
def get_datapack_filename(oar, in_feature):
    return "datapack_" + oar + "_" + in_feature + "_2018.h5" 

# load dataset and return
def load_dataset(in_dir, in_filename, groupname):
    with h5py.File(os.path.join(in_dir, in_filename), 'r') as file_content:
        target_group = file_content[groupname][()]
        return target_group
