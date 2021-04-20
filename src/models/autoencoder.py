from .config import AE_CONFIG, VAE_CONFIG, SEED
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
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import InputLayer, Dense, Activation, Dropout, BatchNormalization
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from datetime import datetime
# import logging
# logger = logging.getLogger(__name__)


class Encoder(tf.keras.layers.Layer):
    def __init__(self, n_features, encoder_neurons, hidden_activation):
        super(Encoder, self).__init__()
        self.encoder_neurons = encoder_neurons
        self.n_features = n_features
        self.hidden_activation = hidden_activation
    
        # capas encoder
        self.input_layer = InputLayer(input_shape=(self.n_features,))
        self.encoder_layers = []
        self.batch_layers = []
        for neurons in self.encoder_neurons:
            self.encoder_layers.append(Dense(
                neurons, 
                activation=self.hidden_activation,
                #activity_regularizer=tf.keras.regularizers.l2(),
                name = F'encoder_{neurons}'))
            self.batch_layers.append(BatchNormalization())

    def call(self, inputs): # , training=None, mask=None
        x = self.input_layer(inputs)
        # Hidden layers
        for i, neurons in enumerate(self.encoder_neurons):
            x = self.encoder_layers[i](x)
            x = self.batch_layers[i](x)
            # x = Dropout(0.3)(x)
        return x


class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class EncoderVAE(tf.keras.Model):
    def __init__(self, n_features, latent_dim, encoder_neurons, hidden_activation):
        super(EncoderVAE, self).__init__()
        self.encoder_neurons = encoder_neurons
        self.n_features = n_features
        self.hidden_activation = hidden_activation
        self.latent_dim = latent_dim    
    
        # capas encoder
        self.input_layer = InputLayer(input_shape=(self.n_features,))
        self.encoder_layers = []
        self.batch_layers = []
        for neurons in self.encoder_neurons:
            self.encoder_layers.append(Dense(
                neurons, 
                activation=self.hidden_activation,
                #activity_regularizer=tf.keras.regularizers.l2(),
                name = F'encoder_{neurons}'))
            self.batch_layers.append(BatchNormalization())
        self.dense_mean = Dense(latent_dim)
        self.dense_log_var = Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs): # , training=None, mask=None
        x = self.input_layer(inputs)
        # Hidden layers
        for i, neurons in enumerate(self.encoder_neurons):
            x = self.encoder_layers[i](x)
            x = self.batch_layers[i](x)
            # x = Dropout(0.3)(x)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z

class Decoder(tf.keras.Model):
    def __init__(self, n_features, latent_dim, decoder_neurons,
                    hidden_activation, output_activation,
                    **kwargs):
        super(Decoder, self).__init__(**kwargs)  

        # decoder layers
        self.decoder_layers = []
        self.batch_layers = []
        for neurons in self.decoder_neurons:
            self.decoder_layers.append(Dense(
                neurons, activation=self.hidden_activation, 
                #activity_regularizer=tf.keras.regularizers.l2(),
                name = F'decoder_{neurons}'))
            self.batch_layers.append(BatchNormalization())

        # output layer
        self.decoder_output = Dense(
            self.n_features, activation=self.output_activation, 
            name = F'decoder_output')

    def call(self, inputs): # ,  training=None, mask=None
        y = inputs
        for i, neurons in enumerate(self.decoder_neurons):
            y = self.decoder_layers[i](y)
            y = self.batch_layers[i](y)
            # y = Dropout(0.3)(y)
        return self.decoder_output(y)

class LossAE(tf.keras.layers.Layer):
    """
    Función de error custom: suma del término de reconstrucción y 
    el término de regularización KL divergence
    """
    def __init__(self, n_features, loss, **kwargs):
        super(LossAE, self).__init__(**kwargs)
        self.n_features = n_features
        self.loss = loss

    def call(self, inputs, training=None, mask=None):
        y_true, y_pred = inputs
        # Error de reconstrucción
        reconstruction_loss = tf.reduce_mean(self.loss(y_true, y_pred))
        # reconstruction_loss *= self.n_features
        return reconstruction_loss

class AE(tf.keras.Model):
    def __init__(
        self, encoder_neurons, decoder_neurons, output_dim,
        hidden_activation, output_activation, **kwargs
        ):
        super(AE, self).__init__(**kwargs)
        self.n_features_ = output_dim
        self.encoder_neurons = encoder_neurons
        self.decoder_neurons = decoder_neurons
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.latent_dim = self.encoder_neurons[-1]
        # self.loss = loss

        # Capas del AE
        self.encoder = Encoder(self)
        self.decoder = Decoder(self)
        self.ae_loss = LossAE(self)

    def call(self, inputs): # , training=None, mask=None
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

    def train_step(self, data):
        with tf.GradientTape() as tape:
            # Propagación hacia delante
            reconstruction = self.call(data)

            loss = self.ae_loss([data, reconstruction])

        # Retro-propagación
        grads = tape.gradient(loss, self.trainable_weights)
        
        # Actualización de los pesos
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        return {"loss": loss}


# class NormalDistribution(tf.keras.Model):
#     def __init__(self, latent_dim, **kwargs):
#         super(NormalDistribution, self).__init__(**kwargs)
#         self.latent_dim = latent_dim
        
#         # Capas
#         self.z_mean = Dense(self.latent_dim, name="z_mean", activation='linear')
#         self.z_log_var = Dense(self.latent_dim, name="z_log_var", activation='linear')

#     def call(self, inputs, training=None, mask=None):
#         z_mean = self.z_mean(inputs)
#         z_log_var = self.z_log_var(inputs)
#         return z_mean, z_log_var


# class Reparametrize(tf.keras.layers.Layer):
#     """
#     Usa (z_mu, z_log_var) para muestrear z (el vector que 
#     representa el dígito codificado)
#     """
#     def call(self, inputs):
#         z_mu, z_log_var = inputs
        
#         # Extraemos dimensiones (del batch y del espacio codificado)
#         batch = tf.shape(z_mu)[0]
#         dim = tf.shape(z_mu)[1]
        
#         # Muestreamos de una distribucion normal ϵ: con dimensiones (batch, dim_spacio_codificado)
#         epsilon = tf.random.normal(shape=(batch, dim))
        
#         # Transformamos log(σ^2) en σ 
#         sigma = tf.exp(0.5 * z_log_var)

#         return z_mu + sigma * epsilon


# class LossVAE(tf.keras.layers.Layer):
#     """
#     Función de error custom: suma del término de reconstrucción y 
#     el término de regularización KL divergence
#     """
#     def __init__(self, n_features, loss, **kwargs):
#         super(LossVAE, self).__init__(**kwargs)
#         self.n_features = n_features
#         self.loss = loss

#     def call(self, inputs, training=None, mask=None):
#         y_true, y_pred, z_mean, z_log_var = inputs
        
#         # Error de reconstrucción
#         reconstruction_loss = tf.reduce_mean(self.loss(y_true, y_pred))
#         # reconstruction_loss *= self.n_features
        
#         # Divergencia KL
#         kl_loss = tf.exp(z_log_var) + tf.square(z_mean) - 1. - z_log_var
#         kl_loss = tf.reduce_mean(0.5*tf.reduce_sum(kl_loss, axis=-1))
    
#         return reconstruction_loss + kl_loss
        

class VAE(tf.keras.Model):
    """
    VAE class; Encoder + Decoder
    """
    def __init__(self, encoder_neurons, decoder_neurons, latent_dim, output_dim, 
                    hidden_activation, output_activation,
                    # loss,
                    **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.n_features_ = output_dim
        self.encoder_neurons = encoder_neurons
        self.decoder_neurons = decoder_neurons
        self.latent_dim = latent_dim
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        # self.loss = loss

        # Capas del VAE
        self.encoder = EncoderVAE(self)
        # self.vae_loss = LossVAE(self)
        # self.normal_distribution = NormalDistribution(self)
        # self.reparametrize = Reparametrize(self)
        self.decoder = Decoder(self)

    def call(self, inputs): # , training=None, mask=None
        z_mean, z_log_var, z = self.encoder(inputs)
        # z_mean, z_log_var = self.normal_distribution(encoded)
        # z = self.reparametrize([z_mean, z_log_var])
        decoded = self.decoder(z)
        # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        return decoded # (z_mean, z_log_var, z), 

    # def train_step(self, data):
    #     with tf.GradientTape() as tape:
    #         # Propagación hacia delante
    #         (z_mean, z_log_var, _), reconstruction = self.call(data)
    #         total_loss = self.vae_loss([data, reconstruction, z_mean, z_log_var])
            
    #     # Retro-propagación
    #     grads = tape.gradient(total_loss, self.trainable_weights)
        
    #     # Actualización de los pesos
    #     self.optimizer.apply_gradients(zip(grads, self.trainable_weights)) 
    #     return {"loss": total_loss}



def build_ae_model(
    encoder_neurons,
    decoder_neurons,
    hidden_activation,
    output_activation,
    learning_rate=0.001,
    loss= AE_CONFIG['LOSS'],
    input_dim = AE_CONFIG['INPUT_DIM'],
    ):
    """
    Build Autoencoder model
    """

    model = AE(
        encoder_neurons = encoder_neurons,
        decoder_neurons = decoder_neurons,
        output_dim=input_dim,
        hidden_activation=hidden_activation,
        output_activation=output_activation,
        loss = loss
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss = loss)

    return model


def fit_ae_model(
    model,
    name : str,
    X_train, 
    validation_data : tuple,
    restore_best_weights = True,
    save_best_only=True,
    verbose : int =0,
    class_weight = None,
    mode : str = AE_CONFIG['MODE'],
    monitor : str = AE_CONFIG['MONITOR'],
    patience : int = AE_CONFIG['PATIENCE'],
    epochs : int = AE_CONFIG['EPOCHS'],
    batch_size : int = AE_CONFIG['BATCH_SIZE'],
    ) -> None:
    """
    Fit Autoencoder model

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

    
    history = model.fit(X_train,
                        epochs=epochs, 
                        batch_size=batch_size,
                        validation_data=validation_data,
                        callbacks=[callbacks],
                        verbose=verbose
                        )
    return history



def build_vae_model(
    encoder_neurons,
    decoder_neurons,
    hidden_activation,
    output_activation,
    learning_rate=0.001,
    loss= VAE_CONFIG['LOSS'],
    input_dim = VAE_CONFIG['INPUT_DIM'],
    latent_dim=VAE_CONFIG['LATENT_DIM']
    ):
    """
    Build Variational Autoencoder model
    """
    reset_random_seeds()

    model = VAE(
        encoder_neurons = encoder_neurons,
        decoder_neurons = decoder_neurons,
        latent_dim=latent_dim,
        output_dim=input_dim,
        hidden_activation=hidden_activation,
        output_activation=output_activation
        # loss = loss,
    )

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss = loss)

    return model


def fit_vae_model(
    model,
    name : str,
    X_train, 
    validation_data,
    restore_best_weights = True,
    save_best_only=True,
    verbose : int =0,
    class_weight = None,
    mode : str = VAE_CONFIG['MODE'],
    monitor : str = VAE_CONFIG['MONITOR'],
    patience : int = VAE_CONFIG['PATIENCE'],
    epochs : int = VAE_CONFIG['EPOCHS'],
    batch_size : int = VAE_CONFIG['BATCH_SIZE'],
    ) -> None:
    """
    Fit Variational Autoencoder model
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

    
    history = model.fit(X_train, X_train,
                        epochs=epochs, 
                        batch_size=batch_size,
                        validation_data=validation_data,
                        callbacks=callbacks,
                        verbose=verbose
                        )

    return history

    

    