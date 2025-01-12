import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class Deep_Learning:
    def __init__(self):
        """
        Initialize Deep Learning Model Class.
        """
        self.df = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initialized Deep Learning Model Class.")
    # Check Stationarity
    def check_stationarity(df, timeseries):
        result = adfuller(timeseries)
        print("ADF Statistic:", result[0])
        print("p-value:", result[1])
        if result[1] <= 0.05:
            print("The data is stationary.")
            df['Sales_diff'] = df['Sales'].diff().dropna()
        else:
            print("The data is not stationary.")
            df['Sales_diff'] = df['Sales']
        return df
            