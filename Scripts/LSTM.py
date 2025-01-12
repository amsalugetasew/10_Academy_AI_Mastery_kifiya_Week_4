import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

class Deep_Learning:
    def __init__(self):
        """
        Initialize Deep Learning Model Class.
        """
        self.df = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initialized Deep Learning Model Class.")
    # Check Stationarity
    def check_stationarity(self, df, timeseries):
        self.logger.info("Starting Checking Whether the time series data is stationary or not")
        result = adfuller(timeseries)
        print("ADF Statistic:", result[0])
        print("p-value:", result[1])
        if result[1] <= 0.05:
            print("The data is stationary.")
            df['Sales_diff'] = df['Sales'].diff().dropna()
        else:
            print("The data is not stationary.")
            df['Sales_diff'] = df['Sales']
        self.logger.info("Successfully Checking Whether the time series data is stationary or not")
        return df
    # Create lag features
    def create_supervised_data(self, series, n_lags=5):
        self.logger.info("Creating supervised data with n_lags=%d", n_lags)
        X, y = [], []
        for i in range(len(series) - n_lags):
            X.append(series[i:i + n_lags])
            y.append(series[i + n_lags])
        self.logger.info("Successfully Created supervised data with n_lags=%d", n_lags)
        return np.array(X), np.array(y)
    # Traing LSTM Model
    def LSTM_model(self,X, y, n_lags):
        self.logger.info("Starting Train LSTM model")
        # Build the LSTM model
        model = Sequential([
            Input(shape=(n_lags, 1)),  # Define input shape explicitly
            LSTM(50, activation='relu', return_sequences=True),
            LSTM(50, activation='relu'),
            Dense(1)
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='mse')
        # model.compile(optimizer=tf.keras.optimizers.Adam(clipvalue=1.0), loss='mse')


        # Train the model
        # model.fit(X, y, epochs=20, batch_size=32, verbose=2)
        model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
        self.logger.info("Successfully Completing Train LSTM model")
        return model