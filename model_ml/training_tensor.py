# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
from __future__ import division
from datetime import datetime, timedelta,date
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pickle

import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam 
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.layers import LSTM
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler

from keras.preprocessing.sequence import TimeseriesGenerator

from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

import warnings
warnings.filterwarnings("ignore")

df_sales = pd.read_csv('/mnt/c/Users/gregoire.jan/OneDrive - Accenture/Documents/Projects/aip/salesforecastaip/data/raw/train.csv')

#represent month in date field as its first day
df_sales['date'] = pd.to_datetime(df_sales['date'])
df_sales['date'] = df_sales['date'].dt.year.astype('str') + '-' + df_sales['date'].dt.month.astype('str') + '-01'
df_sales['date'] = pd.to_datetime(df_sales['date'])

#groupby date and sum the sales
df_sales = df_sales.groupby('date').sales.sum().reset_index()

# Split train
trainset=df_sales[:-6].set_index('date')

# Scale train set
scaler = MinMaxScaler()
scaler.fit(trainset)
scaled_train_data = scaler.transform(trainset)

# Train LSTM
n_input = 12
n_features= 1
generator = TimeseriesGenerator(scaled_train_data, scaled_train_data, length=n_input, batch_size=1)

lstm_model = Sequential()
lstm_model.add(LSTM(200, activation='relu', return_sequences=True, input_shape=(n_input, n_features)))
lstm_model.add(LSTM(200, activation='relu', return_sequences=True))
lstm_model.add(LSTM(200, activation='relu'))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')


lstm_model.summary()

lstm_model.fit_generator(generator,epochs=25)

# Output model
filename = 'salesforecast_lstm.bin'
pickle.dump(lstm_model, open(filename, 'wb'))

