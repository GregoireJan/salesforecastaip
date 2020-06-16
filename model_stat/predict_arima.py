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

# from pyramid.arima import auto_arima
import pmdarima as pm
from pmdarima import model_selection

import warnings
warnings.filterwarnings("ignore")

df_sales = pd.read_csv('/mnt/c/Users/gregoire.jan/OneDrive - Accenture/Documents/Projects/aip/salesforecastaip/data/raw/train.csv')

#represent month in date field as its first day
df_sales['date'] = pd.to_datetime(df_sales['date'])
df_sales['date'] = df_sales['date'].dt.year.astype('str') + '-' + df_sales['date'].dt.month.astype('str') + '-01'
df_sales['date'] = pd.to_datetime(df_sales['date'])

#groupby date and sum the sales
df_sales = df_sales.groupby('date').sales.sum().reset_index()


stepwise_model = pm.auto_arima(df_sales[:-6].sales.values, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)

print(stepwise_model.aic())

stepwise_model.fit(df_sales[:-6].sales.values)

stepwise_model.predict(n_periods=6)

