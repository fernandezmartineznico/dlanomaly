from .config import DNN_CONFIG, SEED
from src.utils import reset_random_seeds

import os
os.environ['PYTHONHASHSEED']=str(SEED)
import numpy as np
np.random.seed(SEED)
np.set_printoptions(precision=4)
import random as python_random
python_random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.utils import class_weight

from datetime import datetime
import logging
logger = logging.getLogger(__name__)


def build_dnn_model(
    n_layers=1, 
    n_neurons=32,
    learning_rate=0.001,
    hidden_activation="relu",
    output_activation = "sigmoid",
    dropout=False,
    batchnorm=False,
    output_bias=None,
    input_dim = DNN_CONFIG['INPUT_DIM'],
    loss=DNN_CONFIG['LOSS'], 
    metrics=DNN_CONFIG['METRICS']
    ) -> None:
    """
    To Do
    """
    reset_random_seeds()

    opt = tf.keras.optimizers.Adam(lr=learning_rate)
    model = Sequential()
    model.add(Dense(n_neurons, input_shape=(input_dim,), name = "dense_input"))
    if batchnorm:
        model.add(BatchNormalization(name = "BN_input"))
        model.add(Activation(hidden_activation))
    if dropout:
        model.add(Dropout(0.3, name = "DO_input"))    

    # Add as many hidden layers as specified in n_layers
    for i in range(n_layers):
        # Layers have n_neurons neurons        
        model.add(Dense(n_neurons, name = F"dense_{i}"))
        if batchnorm:
            model.add(BatchNormalization(name = F"BN_{i}"))
        model.add(Activation(hidden_activation))
        if dropout:
            model.add(Dropout(0.3, name = F"DO_{i}"))

    # final layer
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    model.add(Dense(
        1, activation=output_activation, bias_initializer=output_bias, 
        name = "dense_output"))    

    #compile model
    model.compile(
        optimizer=opt, 
        loss=loss, 
        metrics=metrics)

    return model


def fit_dnn_model(
    model,
    name : str,
    X_train, 
    y_train,
    validation_data : tuple,
    restore_best_weights = True,
    save_best_only=True,
    verbose : int =0,
    class_weight = None,
    mode : str = DNN_CONFIG['MODE'],
    monitor : str = DNN_CONFIG['MONITOR'],
    patience : int = DNN_CONFIG['PATIENCE'],
    epochs : int =DNN_CONFIG['EPOCHS'],
    batch_size : int =DNN_CONFIG['BATCH_SIZE']
    ) -> None:
    """
    To Do
    """
    file_name = name + datetime.now().strftime('%Y%m%d')

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=patience,
            monitor=monitor,
            mode=mode,
            restore_best_weights = restore_best_weights),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=F'..{os.sep}models{os.sep}logs{os.sep}{file_name}.h5',
            save_best_only=save_best_only)
    ]

    history = model.fit(
        X_train, y_train, 
        validation_data=validation_data,
        batch_size=batch_size,
        epochs=epochs, 
        verbose=verbose, 
        callbacks=[callbacks],
        class_weight = class_weight)
    
    return history


