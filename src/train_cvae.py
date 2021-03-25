################################
########## IMPORTS #############
################################

#custom functions
import os, sys
sys.path.insert(0, '..')
from src.models.config import SEED
from src.utils import performance_rank_df, performance_rank_n, performance_rank_f1_opt
from src.utils import plot_precision_recall, plot_loss, plot_auc, plot_history, plot_metrics, plot_rank
from src.utils import plot_label_clusters, plot_label_clusters_vae, plot_label_clusters_cvae
from src.utils import save_report_json, save_report_pandas_to_csv
from src.utils import save_model_joblib, save_model_parameters_pkl, save_model_keras
from src.utils import reset_random_seeds
from pickle import dump

import warnings
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import pyplot
mpl.rcParams['figure.figsize'] = (12, 4)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
warnings.filterwarnings('ignore')
import seaborn as sns

import os
os.environ['PYTHONHASHSEED']=str(SEED)
import numpy as np
np.random.seed(SEED)
np.set_printoptions(precision=4)
import random as python_random
python_random.seed(SEED)
import tensorflow as tf
tf.keras.backend.clear_session()
tf.random.set_seed(SEED)
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, BatchNormalization, Lambda
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras import utils

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split
from sklearn import metrics

print("# TF specs")
print("TensorFlow version: {}".format(tf.__version__))
print("TensorFlow keras version: {}".format(tf.keras.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))


################################
########## DATA PREP ###########
################################

# # Read Data
df = pd.read_csv('../data/raw/creditcard.csv') #pd.read_csv('https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv')
neg, pos = np.bincount(df['Class'])
total = neg + pos
pos / total

# # Data pre-processing
y=df['Class']
df_X=df.drop('Class',1)

df_X['Amount_log'] = np.log(df_X['Amount'] + 1)
df_X=df_X.drop(['Amount', 'Time'],1)

le= LabelEncoder().fit(y)
encoded_Y = le.transform(y)  # convert categorical labels to integers
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = utils.to_categorical(encoded_Y)
X_train, X_test, y_train, y_test = train_test_split(df_X, dummy_y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)

# # Conditional Variational Autoencoder
# With the CVAE, we can ask the model to recreate data (synthetic data) for a particular label.
# configure our pipeline
pipeline_st = Pipeline([('scaler', StandardScaler())])

# get normalization parameters by fitting to the training data
pipeline_st.fit(X_train)

# transform the training and validation data with these parameters
X_train_st = pipeline_st.transform(X_train)
X_val_st = pipeline_st.transform(X_val)
X_test_st = pipeline_st.transform(X_test)

scaler_name = "standardscaler"
filepath_scaler =F'..{os.sep}models{os.sep}{scaler_name}.pkl'
dump(pipeline_st, open(filepath_scaler, 'wb'))

print("# Dataset shapes for (Train, Validation, Test)")
print("X shape: ", X_train.shape, X_val.shape, X_test.shape)
print("Y shape: ", y_train.shape, y_val.shape, y_test.shape)


#################################
########## CVAE MODEL ###########
#################################

from sklearn.utils import class_weight
weights = dict(enumerate(class_weight.compute_class_weight(
    'balanced', np.unique(y_train[:,1]), y_train[:,1])))

print('# Balancing weights to: ', weights)

## Define Hyperparmeters
n_z = 5 # latent space size
encoder_dim_1 = decoder_dim_2 = 20 # dim of encoder hidden layer
encoder_dim_2 = decoder_dim_1 = 10 # dim of decoder hidden layer
decoder_out_dim = X_train.shape[1] # dim of decoder output layer
activ = tf.nn.leaky_relu
n_x = X_train.shape[1]
n_y = y_train.shape[1]

## Explicitely define shape for the encoder
X = Input(shape=(n_x,))
label = Input(shape=(n_y,))

## Concatenate the input and the label for the CVAE
inputs = tf.concat([X, label], 1, name='concat_input')
encoder_h1 = Dense(
  encoder_dim_1, activation=activ, 
  # activity_regularizer=tf.keras.regularizers.l2(),
  name='encoder_1')(inputs)
# bn1 = BatchNormalization(name='bn_1')(encoder_h1)
encoder_h2 = Dense(
  encoder_dim_2, activation=activ, 
  # activity_regularizer=tf.keras.regularizers.l2(), 
  name='encoder_2')(encoder_h1)
# bn2 = BatchNormalization(name='bn_2')(encoder_h2)

mu = Dense(n_z, activation='linear', name='mu')(encoder_h2)
l_sigma = Dense(n_z, activation='linear', name='sigma')(encoder_h2)

class Sampling(tf.keras.layers.Layer):
  def __init__(self):
      super(Sampling, self).__init__()

  def build(self, input_shape):
      _, sigma_shape = input_shape
      self.sigma_shape = (sigma_shape[-1], )

  def call(self, inputs):
      mu, sigma = inputs
      
      # Return sampling as before
      eps = K.random_normal(self.sigma_shape, mean=0., stddev=1.)
      return mu + K.exp(sigma / 2) * eps

# Sampling latent space
# z = Lambda(sample_z, output_shape = (n_z, ))([mu, l_sigma])
z = Sampling()([mu, l_sigma])
# merge latent space with label
zc = tf.concat([z, label], 1, name='concat_latent')

decoder_hidden_1 = Dense(
    decoder_dim_1, activation=activ, 
    # activity_regularizer=tf.keras.regularizers.l2(),
    name='decoder_1')
decoder_hidden_2 = Dense(
    decoder_dim_2, activation=activ, 
    # activity_regularizer=tf.keras.regularizers.l2(),
    name='decoder_2')
decoder_out = Dense(decoder_out_dim, activation=None, name='decoder_output')

h_p_1 = decoder_hidden_1(zc)
# h_p_1_bn = BatchNormalization(name='bn_3')(h_p_1)
h_p_2 = decoder_hidden_2(h_p_1)
# h_p_2_bn = BatchNormalization(name='bn_4')(h_p_2)
outputs = decoder_out(h_p_2)

# instantiate the keras model class API
cvae = Model([X, label], outputs)
encoder = Model([X, label], mu)

d_in = Input(shape=(n_z+n_y,))
d_h1 = decoder_hidden_1(d_in)
# d_h1_bn = BatchNormalization(name='bn_3')(d_h1)
d_h2 = decoder_hidden_2(d_h1)
# d_h2_bn = BatchNormalization(name='bn_4')(d_h2)
d_out = decoder_out(d_h2)
decoder = Model(d_in, d_out)

## Model will be trained on vae loss
def vae_loss(y_true, y_pred):
  recon = K.sum(K.mean(K.square(y_pred - y_true), axis=-1))
  kl = 0.5 * K.sum(K.exp(l_sigma) + K.square(mu) - 1. - l_sigma, axis=-1)
  return recon + kl

def KL_loss(y_true, y_pred):
  return (0.5 * K.sum(K.exp(l_sigma) + K.square(mu) - 1. - l_sigma, axis=1))

def recon_loss(y_true, y_pred):
  return K.sum(K.mean(K.square(y_pred - y_true), axis=-1))

kl_loss = -0.5 * tf.reduce_mean(l_sigma - tf.square(mu) - tf.exp(l_sigma) + 1)
cvae.add_loss(kl_loss)

## Compile function
# cvae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=vae_loss, metrics = [KL_loss, recon_loss])
cvae.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
    loss=recon_loss,
    metrics = [recon_loss])

name = 'cvae'
file_name = name + datetime.now().strftime('%Y%m%d')

callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=10,
            monitor="val_loss",
            restore_best_weights = True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=F'..{os.sep}models{os.sep}logs{os.sep}{file_name}.h5',
            save_best_only=True)
    ]

# compile and fit
cvae_hist = cvae.fit([X_train_st, y_train], X_train_st, 
                     validation_data = ([X_val_st, y_val], X_val_st),
                     batch_size=200, epochs=100,
                     callbacks = callbacks, 
                     verbose = 0
                     ).history

# save encoder model
encoder_name = "cvae-encoder"
file_name = encoder_name + datetime.now().strftime('%Y%m%d')
encoder.save(F'..{os.sep}models{os.sep}logs{os.sep}{file_name}.h5')

## Plot loss
# Check plots - Loss
plt.plot(cvae_hist['loss'])
plt.plot(cvae_hist['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right');

scores_cvae_st = np.mean(np.power(X_test_st - cvae.predict([X_test_st, np.asarray([[1., 0.]]*X_test_st.shape[0])]), 2), axis=1)
fpr, tpr, thresholds = metrics.roc_curve(np.array(y_test[:,1]), scores_cvae_st, pos_label=1)
auc = metrics.auc(fpr, tpr)
print("# AUC TEST CVAE: ",auc)

df_pf_cvae_st = performance_rank_df(y_test[:,1], scores_cvae_st)
print(performance_rank_n(df_pf_cvae_st))

plot_label_clusters_cvae(encoder, X_val_st, y_val[:,1])


#################################
########## CVAE MERGED ##########
#################################

z_mean_train = encoder.predict([X_train_st, np.asarray([[1., 0.]]*X_train_st.shape[0])])
z_mean_val = encoder.predict([X_val_st, np.asarray([[1., 0.]]*X_val_st.shape[0])])
z_mean_test = encoder.predict([X_test_st, np.asarray([[1., 0.]]*X_test_st.shape[0])])

left_branch_input = Input(shape=(29,), name='Left_input')
left_branch_output = Dense(64, activation='relu')(left_branch_input)

right_branch_input = Input(shape=(5,), name='Right_input')
right_branch_output = Dense(10, activation='relu')(right_branch_input)

concat = tf.concat([left_branch_output, right_branch_output], 1,name='Concatenate')
final_model_output = Dense(1, activation='sigmoid')(concat)
final_model = Model(inputs=[left_branch_input, right_branch_input], outputs=final_model_output,
                    name='Final_output')

final_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(name='auc')])

name = 'cvae_dnn_merged'
file_name = name + datetime.now().strftime('%Y%m%d')

final_model_history = final_model.fit([X_train_st, z_mean_train], y_train[:,1],
    validation_data= ([X_val_st, z_mean_val], y_val[:,1]),
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=5, monitor="val_auc", mode = 'max', restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=F'..{os.sep}models{os.sep}logs{os.sep}{file_name}.h5',save_best_only=True)],
    epochs = 100,
    batch_size = 128,
    verbose=0,
    class_weight = weights
    )

# ## Plot loss
# # Check plots - Loss
plt.plot(final_model_history.history['loss'])
plt.plot(final_model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right');

fpr, tpr, thresholds = metrics.roc_curve(y_test[:,1], final_model.predict([X_test.to_numpy(), z_mean_test])[:,0], pos_label=1)
print("# AUC TEST FINAL MODEL (CVAE+DNN merged): ",metrics.auc(fpr, tpr))


df_pf_merged = performance_rank_df(y_test[:,1], final_model.predict([X_test.to_numpy(), z_mean_test])[:,0], if_score = False)
print(performance_rank_n(df_pf_merged))



