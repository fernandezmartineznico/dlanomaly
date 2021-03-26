################################
########## IMPORTS #############
################################

#custom functions
import os, sys
sys.path.insert(0, '..')
from src.models.config import SEED
from src.utils import performance_rank_df, performance_rank_n, performance_rank_f1_opt
from pickle import load
import warnings
from datetime import datetime
import matplotlib.pyplot as plt

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
from tensorflow.keras.models import load_model
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



###############################
########## DATA PREP ###########
################################

# # Read Data
df_X = pd.read_csv('../data/prod/creditcard_prod.csv') #pd.read_csv('https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv')

# # Data pre-processing
df_X['Amount_log'] = np.log(df_X['Amount'] + 1)
df_X=df_X.drop(['Amount', 'Time'],1)

# load scaler
scaler_name = "standardscaler"
filepath_scaler =F'..{os.sep}models{os.sep}{scaler_name}.pkl'
pipeline_st = load(open(filepath_scaler, 'rb'))
X_pred_st = pipeline_st.transform(df_X)

print("# Data shape")
print(X_pred_st.shape)


################################
########## PREDICT #############
################################

# load encoder 
encoder_name = "cvae-encoder20210325"
encoder = load_model(
    F'..{os.sep}models{os.sep}logs{os.sep}{encoder_name}.h5', 
    custom_objects={'leaky_relu': tf.nn.leaky_relu}, 
    compile=False)

# latent projection from CVAE
z_mean_pred = encoder.predict([X_pred_st, np.asarray([[1., 0.]]*X_pred_st.shape[0])])

# load final model
final_model_name = "cvae_dnn_merged20210325"
final_model = load_model(
    F'..{os.sep}models{os.sep}logs{os.sep}{final_model_name}.h5', 
    compile=False)

pred = final_model.predict([X_pred_st, z_mean_pred])[:,0]
print("# Predictions: ", pred[:5])

# Write AUC to a file
report_name = "predictions" #"predictions-" + datetime.now().strftime('%Y%m%d')
np.savetxt(F"../reports/predictions/{report_name}.txt", pred, delimiter=',')

plt.style.use('ggplot')
plt.hist(pred, bins=50)
plt.axvline(pred.mean(), color='k', linestyle='dashed', linewidth=1)
min_ylim, max_ylim = plt.ylim()
plt.text(pred.mean()*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(pred.mean()))
plt.title("predictions-" + datetime.now().strftime('%Y%m%d'))

report_figure_name = "histogram_predictions" #"histogram_predictions-" + datetime.now().strftime('%Y%m%d')
plt.savefig(F"../reports/figures/{report_figure_name}.png",dpi=120)
