# coding: utf-8

from base.base_data_loader import BaseDataLoader
from utils.data_utils import get_datapack_filename, load_dataset

class ProstateDistDvhDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(ProstateDistDvhDataLoader, self).__init__(config)

        # (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()
        #print "datapack_" + config.data_loader.organ + "_" + config.data_loader.x_name + "_2018.h5" 
        filename = get_datapack_filename(str(config.data_loader.organ), str(config.data_loader.x_name))
        self.X = load_dataset(config.data_loader.h5py_dir, filename, config.data_loader.x_groupname)
        self.y = load_dataset(config.data_loader.h5py_dir, filename, config.data_loader.y_groupname)
        config.data_loader.input_shape = self.X.shape[1:]
        
        # train-test / train-val-test        

    # def set_train_test_val(tr_split=0.8, val_split=0.2):
    def get_data(self):
        return self.X, self.y
        
    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test

