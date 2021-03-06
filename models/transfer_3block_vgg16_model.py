# coding: utf-8
from base.base_model import BaseModel
from utils.learning_utils import recall, precision, fbeta_score

#from keras import backend as K
#from keras import utils as np_utils
from keras.models import Model
from keras.optimizers import *
from keras.applications import VGG16

from keras.layers import Input, Dropout, Conv2D, MaxPooling2D, Dense, Flatten, Activation


class Transfer3BlockVGG16Model(BaseModel):
    def __init__(self, config):
        super(Transfer3BlockVGG16Model, self).__init__(config)
        self.build_model()

    def build_model(self):
        # If you want to specify input tensor
        input_tensor = Input(shape=self.config.data_loader.input_shape)
        vgg_model = VGG16(weights='imagenet',
                          include_top=False,
                          input_tensor=input_tensor)
        
        # Creating dictionary that maps layer names to the layers
        layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])

        # Getting output tensor of the last VGG layer that we want to include
        x = layer_dict['block3_pool'].output
    
        # Stacking a new simple convolutional network on top of it    
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.25)(x) #(space['drop_out'])(x)
        x = Dense(1, activation='sigmoid')(x)

        # Creating new model. Please note that this is NOT a Sequential() model.
        self.model = Model(inputs=vgg_model.input, outputs=x)
        
        # self.model.summary()
        # print custom_model.layers[15]

        # Make sure that the pre-trained bottom layers are not trainable
        for layer in self.model.layers[2:11]:
            layer.trainable = False
        for layer in self.model.layers[0:2]:
            layer.trainable = True
        for layer in self.model.layers[11:]:
            layer.trainable = True

        # Do not forget to compile it
        self.model.compile(
            loss=self.config.model.loss,
            optimizer=Adam(lr=self.config.model.learning_rate),
            metrics=['accuracy', recall, precision, fbeta_score])


# batch_size=10, weights=None):
# space={'learning_rate':float({{uniform(0,1)}}),
    #        'drop_out':float({{uniform(0,1)}})}
    
#         self.model.compile(
#               loss='sparse_categorical_crossentropy',
#               optimizer=self.config.model.optimizer,
#               metrics=['accuracy'])
#space['learning_rate']), #'Adam',
