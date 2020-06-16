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
from sklearn.preprocessing import MinMaxScaler


## Read data and model
df_sales = pd.read_csv('/mnt/c/Users/gregoire.jan/OneDrive - Accenture/Documents/Projects/aip/salesforecastaip/data/raw/train.csv')

with open('salesforecast_lstm.bin', 'rb') as file:
    lstm_model = pickle.load(file)


## Prep data for prediction
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


n_input = 12
n_features= 1

lstm_predictions_scaled = list()

batch = scaled_train_data[-n_input:]
current_batch = batch.reshape((1, n_input, n_features))

# Prediction
for i in range(6):   
    lstm_pred = lstm_model.predict(current_batch)[0]
    lstm_predictions_scaled.append(lstm_pred) 
    current_batch = np.append(current_batch[:,1:,:],[[lstm_pred]],axis=1)

lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)
lstm_predictions

