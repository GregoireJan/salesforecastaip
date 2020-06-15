{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": "array([[1126906.96384633],\n       [1032720.93958294],\n       [ 972138.21210527],\n       [ 957892.05993295],\n       [ 941188.53760612],\n       [ 686087.4831211 ]])"
     },
     "metadata": {},
     "execution_count": 2
    }
   ],
   "source": [
    "from __future__ import division\n",
    "from datetime import datetime, timedelta,date\n",
    "import pandas as pd\n",
    "%matplotlib inline\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import pickle\n",
    "from sklearn.preprocessing import MinMaxScaler\n",
    "\n",
    "\n",
    "## Read data and model\n",
    "df_sales = pd.read_csv('/mnt/c/Users/gregoire.jan/OneDrive - Accenture/Documents/Projects/aip/salesforecastaip/data/raw/train.csv')\n",
    "\n",
    "with open('salesforecast_lstm.bin', 'rb') as file:\n",
    "    lstm_model = pickle.load(file)\n",
    "\n",
    "\n",
    "## Prep data for prediction\n",
    "#represent month in date field as its first day\n",
    "df_sales['date'] = pd.to_datetime(df_sales['date'])\n",
    "df_sales['date'] = df_sales['date'].dt.year.astype('str') + '-' + df_sales['date'].dt.month.astype('str') + '-01'\n",
    "df_sales['date'] = pd.to_datetime(df_sales['date'])\n",
    "\n",
    "#groupby date and sum the sales\n",
    "df_sales = df_sales.groupby('date').sales.sum().reset_index()\n",
    "\n",
    "# Split train\n",
    "trainset=df_sales[:-6].set_index('date')\n",
    "\n",
    "# Scale train set\n",
    "scaler = MinMaxScaler()\n",
    "scaler.fit(trainset)\n",
    "scaled_train_data = scaler.transform(trainset)\n",
    "\n",
    "\n",
    "n_input = 12\n",
    "n_features= 1\n",
    "\n",
    "lstm_predictions_scaled = list()\n",
    "\n",
    "batch = scaled_train_data[-n_input:]\n",
    "current_batch = batch.reshape((1, n_input, n_features))\n",
    "\n",
    "# Prediction\n",
    "for i in range(6):   \n",
    "    lstm_pred = lstm_model.predict(current_batch)[0]\n",
    "    lstm_predictions_scaled.append(lstm_pred) \n",
    "    current_batch = np.append(current_batch[:,1:,:],[[lstm_pred]],axis=1)\n",
    "\n",
    "lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)\n",
    "lstm_predictions"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.9-final"
  },
  "orig_nbformat": 2,
  "kernelspec": {
   "name": "python36964bit01763e9be29f414ba19e1fc9f5b6ec3a",
   "display_name": "Python 3.6.9 64-bit"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}