{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": "Model: \"sequential_10\"\n_________________________________________________________________\nLayer (type)                 Output Shape              Param #   \n=================================================================\nlstm_18 (LSTM)               (None, 12, 200)           161600    \n_________________________________________________________________\nlstm_19 (LSTM)               (None, 12, 200)           320800    \n_________________________________________________________________\nlstm_20 (LSTM)               (None, 200)               320800    \n_________________________________________________________________\ndense_10 (Dense)             (None, 1)                 201       \n=================================================================\nTotal params: 803,401\nTrainable params: 803,401\nNon-trainable params: 0\n_________________________________________________________________\nEpoch 1/25\n42/42 [==============================] - 2s 56ms/step - loss: 0.0971\nEpoch 2/25\n42/42 [==============================] - 1s 34ms/step - loss: 0.0733\nEpoch 3/25\n42/42 [==============================] - 1s 35ms/step - loss: 0.0515\nEpoch 4/25\n42/42 [==============================] - 1s 31ms/step - loss: 0.0688\nEpoch 5/25\n42/42 [==============================] - 1s 31ms/step - loss: 0.0432\nEpoch 6/25\n42/42 [==============================] - 1s 32ms/step - loss: 0.0351\nEpoch 7/25\n42/42 [==============================] - 1s 32ms/step - loss: 0.0263\nEpoch 8/25\n42/42 [==============================] - 1s 31ms/step - loss: 0.0146\nEpoch 9/25\n42/42 [==============================] - 1s 31ms/step - loss: 0.0075\nEpoch 10/25\n42/42 [==============================] - 1s 33ms/step - loss: 0.0048\nEpoch 11/25\n42/42 [==============================] - 1s 31ms/step - loss: 0.0078\nEpoch 12/25\n42/42 [==============================] - 1s 32ms/step - loss: 0.0073\nEpoch 13/25\n42/42 [==============================] - 1s 34ms/step - loss: 0.0039\nEpoch 14/25\n42/42 [==============================] - 1s 34ms/step - loss: 0.0058\nEpoch 15/25\n42/42 [==============================] - 1s 32ms/step - loss: 0.0065\nEpoch 16/25\n42/42 [==============================] - 1s 32ms/step - loss: 0.0118\nEpoch 17/25\n42/42 [==============================] - 1s 32ms/step - loss: 0.0055\nEpoch 18/25\n42/42 [==============================] - 1s 31ms/step - loss: 0.0040\nEpoch 19/25\n42/42 [==============================] - 1s 33ms/step - loss: 0.0047\nEpoch 20/25\n42/42 [==============================] - 1s 33ms/step - loss: 0.0039\nEpoch 21/25\n42/42 [==============================] - 1s 33ms/step - loss: 0.0058\nEpoch 22/25\n42/42 [==============================] - 1s 34ms/step - loss: 0.0037\nEpoch 23/25\n42/42 [==============================] - 1s 34ms/step - loss: 0.0058\nEpoch 24/25\n42/42 [==============================] - 1s 31ms/step - loss: 0.0076\nEpoch 25/25\n42/42 [==============================] - 1s 33ms/step - loss: 0.0036\n"
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
    "\n",
    "import keras\n",
    "from keras.layers import Dense\n",
    "from keras.models import Sequential\n",
    "from keras.optimizers import Adam \n",
    "from keras.callbacks import EarlyStopping\n",
    "from keras.utils import np_utils\n",
    "from keras.layers import LSTM\n",
    "from sklearn.model_selection import KFold, cross_val_score, train_test_split\n",
    "from sklearn.preprocessing import MinMaxScaler\n",
    "\n",
    "from keras.preprocessing.sequence import TimeseriesGenerator\n",
    "\n",
    "from keras.layers import Flatten\n",
    "from keras.layers import TimeDistributed\n",
    "from keras.layers.convolutional import Conv1D\n",
    "from keras.layers.convolutional import MaxPooling1D\n",
    "\n",
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")\n",
    "\n",
    "df_sales = pd.read_csv('/mnt/c/Users/gregoire.jan/OneDrive - Accenture/Documents/Projects/aip/salesforecastaip/data/raw/train.csv')\n",
    "\n",
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
    "# Train LSTM\n",
    "n_input = 12\n",
    "n_features= 1\n",
    "generator = TimeseriesGenerator(scaled_train_data, scaled_train_data, length=n_input, batch_size=1)\n",
    "\n",
    "lstm_model = Sequential()\n",
    "lstm_model.add(LSTM(200, activation='relu', return_sequences=True, input_shape=(n_input, n_features)))\n",
    "lstm_model.add(LSTM(200, activation='relu', return_sequences=True))\n",
    "lstm_model.add(LSTM(200, activation='relu'))\n",
    "lstm_model.add(Dense(1))\n",
    "lstm_model.compile(optimizer='adam', loss='mse')\n",
    "\n",
    "\n",
    "lstm_model.summary()\n",
    "\n",
    "lstm_model.fit_generator(generator,epochs=25)\n",
    "\n",
    "# Output model\n",
    "filename = 'salesforecast_lstm.bin'\n",
    "pickle.dump(lstm_model, open(filename, 'wb'))"
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