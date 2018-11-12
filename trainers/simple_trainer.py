
# coding: utf-8
from base.base_trainer import BaseTrain
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau

import os

class SimpleTrainer(BaseTrain):
    def __init__(self, model, data, config):
        super(SimpleTrainer, self).__init__(model, data, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_callbacks()

    def init_callbacks(self):
        
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.callbacks.checkpoint_dir, '%s-%s-%s-{epoch:02d}-{val_loss:.3f}.hdf5' % (self.config.exp.name, self.config.data_loader.organ, self.config.data_loader.x_name)),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                verbose=self.config.callbacks.checkpoint_verbose,
            )
        )
        
        self.callbacks.append(
            ReduceLROnPlateau(
                monitor=self.config.callbacks.reducelronplateau_monitor,
                min_delta=self.config.callbacks.reducelronplateau_min_delta, 
                patience=self.config.callbacks.reducelronplateau_patience, 
                verbose=self.config.callbacks.reducelronplateau_verbose,
                min_lr=self.config.callbacks.reducelronplateau_min_lr,
            )
        )

        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.callbacks.tensorboard_log_dir,
                write_graph=self.config.callbacks.tensorboard_write_graph,
            )
        )

    def train(self):
        history = self.model.fit(
            self.data[0], self.data[1],
            epochs=self.config.trainer.num_epochs,
            shuffle=self.config.trainer.shuffle,
            verbose=self.config.trainer.verbose_training,
            batch_size=self.config.trainer.batch_size,
            validation_split=self.config.trainer.validation_split,
            callbacks=self.callbacks
        )
        self.loss.extend(history.history['loss'])
        self.acc.extend(history.history['acc'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_acc.extend(history.history['val_acc'])
        
    def train_val(self, train_indexes, test_indexes):
        
        # Fit the model
        history = self.model.fit(
            self.data[0][train_indexes], self.data[1][train_indexes],
            epochs=self.config.trainer.num_epochs,
            verbose=self.config.trainer.verbose_training,
            batch_size=self.config.trainer.batch_size,
            validation_split=self.config.trainer.validation_split
            #,callbacks=self.callbacks
        )
        
        # evaluate the model
        scores = self.model.evaluate(
            self.data[0][test_indexes], self.data[1][test_indexes],
            verbose=self.config.trainer.verbose_training
        )
        
        return scores
