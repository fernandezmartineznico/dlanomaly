import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split


class DataProcessor:
    """
    Class for reading, processing, and writing data 
    """
    def __init__(self):
        pass

    def read_data(self, raw_data_path):
        """Read raw data into DataProcessor."""
        df = pd.read_csv('../data/raw/creditcard.csv')
        return df

    def process_data(self, df, test_size=0.2, stable=True):
        """Process raw data into useful files for model."""
        y = df['Class']
        X = df.drop('Class',1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
        return X_train, X_test, y_train, y_test

    def write_data(self, processed_data_path):
        """Write processed data to directory."""
        return