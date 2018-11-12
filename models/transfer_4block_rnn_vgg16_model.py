# coding: utf-8
from base.base_model import BaseModel
from hyperas.distributions import choice, uniform, conditional
from utils.learning_utils import recall, precision, fbeta_score

#from keras import backend as K
#from keras import utils as np_utils
from keras.models import Model
from keras.optimizers import *
from keras.applications import VGG16

from keras.layers import Input, Dropout, Conv2D, MaxPooling2D, Dense, Flatten, Activation, TimeDistributed, LSTM


class Transfer4BlockRnnVGG16Model(BaseModel):
    def __init__(self, config):
        super(Transfer4BlockRnnVGG16Model, self).__init__(config)
        self.build_model()
        
    def get_single_slice_model(self):
        # If you want to specify input tensor
        # print self.config.data_loader.input_shape[1:] #(90, 70, 3)
        input_tensor = Input(shape=self.config.data_loader.input_shape[1:])
        vgg_model = VGG16(weights='imagenet',
                          include_top=False,
                          input_tensor=input_tensor)
        
        # Creating dictionary that maps layer names to the layers
        layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])

        # Getting output tensor of the last VGG layer that we want to include
        x = layer_dict['block4_pool'].output
    
        # Stacking a new simple convolutional network on top of it    
        x = Flatten()(x)
        # x = Dense(512, activation='relu')(x)
        # x = Dense(128, activation='relu')(x)
        # x = Dropout(0.25)(x) #space['drop_out'])(x)
        # x = Dense(1, activation='sigmoid')(x)

        # Creating new model. Please note that this is NOT a Sequential() model.
        model = Model(inputs=input_tensor, outputs=x)
        
        # self.model.summary()
        # print custom_model.layers[15]

        # Make sure that the pre-trained bottom layers are not trainable
        for layer in model.layers[2:11]:
            layer.trainable = False
        for layer in model.layers[0:2]:
            layer.trainable = True
        # for layer in model.layers[15:]:
        #     layer.trainable = True
            
        return model
    
    def build_model(self):
        # exploration #
        # space={'drop_out':float({{uniform(0,0.5)}})}
        
        # If you want to specify input tensor
        patient_shape = self.config.data_loader.input_shape
        
        # tuple([self.config.data_loader.x_slices]) + self.config.data_loader.input_shape
        patient_input = Input(shape=patient_shape) #input_shape=(5, 224, 224, 3)
        
        # define LSTM model
        encoded_frames = TimeDistributed(self.get_single_slice_model())(patient_input)
        encoded_sequence = LSTM(self.config.data_loader.x_slices,return_sequences=False)(encoded_frames)
        # ,
        #batch_input_shape=5
        # Stacking a new simple convolutional network on top of it    
        hidden_layer = Dense(512, activation='relu')(encoded_sequence)
        hidden_layer = Dense(128, activation='relu')(hidden_layer)
        hidden_layer = Dropout(0.25)(hidden_layer) #space['drop_out'])(x)
        output = Dense(1, activation='sigmoid')(hidden_layer)
        
        # Creating new model. Please note that this is NOT a Sequential() model.
        self.model = Model(inputs=patient_input, outputs=output)
        #Model(inputs=patient_input, outputs=x)
        
        optimizer = Nadam(lr=0.002,
                  beta_1=0.9,
                  beta_2=0.999,
                  epsilon=1e-08,
                  schedule_decay=0.004)
        
        # Do not forget to compile it
        self.model.compile(
            loss=self.config.model.loss,
            optimizer=optimizer, #Adam(lr=self.config.model.learning_rate),
            metrics=[precision, recall, fbeta_score, 'accuracy'])
        
        
        # input_tensor = Input(shape=self.config.data_loader.input_shape) #input_shape=(5, 224, 224, 3)
        # video = Input(shape=(frames, channels, rows, columns))
        
        # hidden_layer = Dense(output_dim=1024, activation="relu")(encoded_sequence)
        # self.model.summary()
        # print custom_model.layers[15]
        
        # cnn = Model(input=vgg_model.input, output=vgg_out)
            
        
      