SEED = 42
import tensorflow as tf

# model params
DNN_CONFIG = {
    'EPOCHS' : 100,
    'BATCH_SIZE' : 2048, # Big given the class imbalancement
    'PATIENCE' : 10,
    'LOSS' : 'binary_crossentropy',
    'METRICS' : [tf.metrics.AUC(name='auc')],
    'MONITOR':'val_auc',
    'MODE' : 'max',
    'INPUT_DIM' : 29
}

AE_CONFIG = {
    'EPOCHS' : 50,
    'BATCH_SIZE' : 32 ,
    'PATIENCE' : 3,
    'LOSS' : tf.losses.mse,
    'METRICS' : [tf.losses.mse],
    'MONITOR':'val_loss',
    'MODE' : 'min',
    'INPUT_DIM' : 29
}

VAE_CONFIG = {
    'EPOCHS' : 50,
    'BATCH_SIZE' : 32 ,
    'PATIENCE' : 3,
    'LOSS' : tf.losses.mse,
    'METRICS' : [tf.losses.mse],
    'MONITOR':'val_loss',
    'MODE' : 'min',
    'INPUT_DIM' : 29,
    'LATENT_DIM' : 2
}

# Grid search parameters
GRID_PARAMETERS = dict(
    n_layers=[0, 1, 2], 
    n_neurons=[32, 64, 128],
    learning_rate=[0.1, 0.01,  0.001]
    )