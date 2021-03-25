import pandas as pd
import numpy as np
import json
import os
import joblib
from datetime import datetime

from src.models.config import SEED
import random as python_random
import tensorflow as tf


import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
mpl.rcParams['figure.figsize'] = (10, 6)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def plot_rank_precision_recall(df):
    """
    To Do
    """
    df.plot(x='Rank', y=['Precision', 'Recall', 'F1_score'])

def plot_precision_recall(df, label, color_n):
    """
    To Do
    """
    plt.plot(df['Recall'], df['Precision'],  
        label='PR ' + label, color=colors[color_n])
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.legend(loc='best')

def plot_loss(history, label, color_n):
    # Use a log scale on y-axis to show the wide range of values.
    plt.semilogy(history.epoch, history.history['loss'],
                color=colors[color_n], label='Train ' + label)
    plt.semilogy(history.epoch, history.history['val_loss'],
                color=colors[color_n], label='Val ' + label,
                linestyle="--")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')

def plot_auc(history, label, color_n):
    plt.plot(history.epoch, history.history['auc'],
                color=colors[color_n], label='Train ' + label)
    plt.plot(history.epoch, history.history['val_auc'],
                color=colors[color_n], label='Val ' + label,
                linestyle="--")
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend(loc='best')

def plot_rank(df, label, color_n = 0, metric = "Recall", top_n = 500):
    df = df[0:top_n].reset_index()
    plt.plot(df.Rank, df[metric],color=colors[color_n], label=metric + '_' + label)
    plt.xlabel('Rank')
    plt.ylabel(metric)
    plt.legend(loc='best')

def plot_history(history, color_n):
    metrics = [m for m in history.history.keys() if not m.startswith('val')]
    for i,metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,i+1)
        plt.plot(history.epoch, history.history[metric], color=colors[color_n], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                color=colors[color_n], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])
        plt.legend(loc  = 'best')

def plot_metrics(histories):
    for n, history in enumerate(histories):
        plot_history(history, n)


def plot_label_clusters(model, data, labels, fig_size=12, scale=3.):
    # Dibujamos un plot 2D the los dígitos y sus clases en el espacio codificado
    encoded = model.encoder(data)
    z_mean, _ = model.normal_distribution(encoded)
    plt.figure(figsize=(fig_size, fig_size))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels, alpha = 0.5)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    # plt.xlim((-scale, scale))
    # plt.ylim((-scale, scale)) 
    plt.show()

def plot_label_clusters_vae(model, data, labels, fig_size=12):
    print(labels.unique())
    colors = []
    for i, val in enumerate(labels.unique()):
        colors.append(sns.color_palette()[i])
    z_mean,_,_ = model.predict(data)
    df_z_mean = pd.DataFrame(z_mean).add_prefix('Z_')
    df_z_mean["label"] = labels
    sns.pairplot(pd.DataFrame(df_z_mean), hue="label", palette=colors)

def plot_label_clusters_cvae(model, data, labels, fig_size=12):
    z_mean = model.predict([data, np.asarray([[1., 0.]]*data.shape[0])])
    df_z_mean = pd.DataFrame(z_mean).add_prefix('Z_')
    df_z_mean["label"] = labels
    sns.pairplot(pd.DataFrame(df_z_mean), hue="label", palette=[sns.color_palette()[0], sns.color_palette()[1]])