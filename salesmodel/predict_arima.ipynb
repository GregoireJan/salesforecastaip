{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": "Performing stepwise search to minimize aic\nFit ARIMA(1,1,1)x(0,1,1,12) [intercept=True]; AIC=914.270, BIC=922.838, Time=0.692 seconds\nFit ARIMA(0,1,0)x(0,1,0,12) [intercept=True]; AIC=923.078, BIC=926.505, Time=0.019 seconds\nFit ARIMA(1,1,0)x(1,1,0,12) [intercept=True]; AIC=914.029, BIC=920.884, Time=0.124 seconds\nFit ARIMA(0,1,1)x(0,1,1,12) [intercept=True]; AIC=913.890, BIC=920.744, Time=0.159 seconds\nFit ARIMA(0,1,0)x(0,1,0,12) [intercept=False]; AIC=921.221, BIC=922.935, Time=0.029 seconds\nFit ARIMA(0,1,1)x(0,1,0,12) [intercept=True]; AIC=925.199, BIC=930.339, Time=0.108 seconds\nFit ARIMA(0,1,1)x(1,1,1,12) [intercept=True]; AIC=913.504, BIC=922.071, Time=0.372 seconds\nFit ARIMA(0,1,1)x(1,1,0,12) [intercept=True]; AIC=914.412, BIC=921.266, Time=0.248 seconds\nFit ARIMA(0,1,1)x(2,1,1,12) [intercept=True]; AIC=913.254, BIC=923.536, Time=1.592 seconds\nNear non-invertible roots for order (0, 1, 1)(2, 1, 1, 12); setting score to inf (at least one inverse root too close to the border of the unit circle: 0.996)\nFit ARIMA(0,1,1)x(1,1,2,12) [intercept=True]; AIC=913.646, BIC=923.927, Time=1.524 seconds\nNear non-invertible roots for order (0, 1, 1)(1, 1, 2, 12); setting score to inf (at least one inverse root too close to the border of the unit circle: 0.996)\nFit ARIMA(0,1,1)x(0,1,2,12) [intercept=True]; AIC=915.642, BIC=924.210, Time=0.883 seconds\nFit ARIMA(0,1,1)x(2,1,0,12) [intercept=True]; AIC=916.292, BIC=924.860, Time=0.378 seconds\nFit ARIMA(0,1,1)x(2,1,2,12) [intercept=True]; AIC=912.025, BIC=924.020, Time=2.081 seconds\nNear non-invertible roots for order (0, 1, 1)(2, 1, 2, 12); setting score to inf (at least one inverse root too close to the border of the unit circle: 0.992)\nFit ARIMA(0,1,0)x(1,1,1,12) [intercept=True]; AIC=911.003, BIC=917.857, Time=0.406 seconds\nFit ARIMA(0,1,0)x(0,1,1,12) [intercept=True]; AIC=911.274, BIC=916.414, Time=0.137 seconds\nFit ARIMA(0,1,0)x(1,1,0,12) [intercept=True]; AIC=911.733, BIC=916.873, Time=0.175 seconds\nFit ARIMA(0,1,0)x(2,1,1,12) [intercept=True]; AIC=910.682, BIC=919.250, Time=1.570 seconds\nNear non-invertible roots for order (0, 1, 0)(2, 1, 1, 12); setting score to inf (at least one inverse root too close to the border of the unit circle: 0.996)\nFit ARIMA(0,1,0)x(1,1,2,12) [intercept=True]; AIC=911.107, BIC=919.675, Time=1.316 seconds\nNear non-invertible roots for order (0, 1, 0)(1, 1, 2, 12); setting score to inf (at least one inverse root too close to the border of the unit circle: 0.996)\nFit ARIMA(0,1,0)x(0,1,2,12) [intercept=True]; AIC=913.027, BIC=919.882, Time=0.317 seconds\nFit ARIMA(0,1,0)x(2,1,0,12) [intercept=True]; AIC=913.623, BIC=920.477, Time=0.349 seconds\nFit ARIMA(0,1,0)x(2,1,2,12) [intercept=True]; AIC=908.828, BIC=919.110, Time=2.021 seconds\nFit ARIMA(1,1,0)x(2,1,2,12) [intercept=True]; AIC=911.267, BIC=923.262, Time=1.701 seconds\nFit ARIMA(1,1,1)x(2,1,2,12) [intercept=True]; AIC=912.778, BIC=926.487, Time=4.007 seconds\nTotal fit time: 20.239 seconds\n908.8281399363067\n"
    },
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": "array([1183802.38117097, 1018368.43127087,  932724.46913935,\n        897965.7777744 ,  924471.43868682,  691611.13195069])"
     },
     "metadata": {},
     "execution_count": 6
    }
   ],
   "source": [
    "from __future__ import division\n",
    "from datetime import datetime, timedelta,date\n",
    "import pandas as pd\n",
    "%matplotlib inline\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "\n",
    "# from pyramid.arima import auto_arima\n",
    "import pmdarima as pm\n",
    "from pmdarima import model_selection\n",
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
    "\n",
    "stepwise_model = pm.auto_arima(df_sales[:-6].sales.values, start_p=1, start_q=1,\n",
    "                           max_p=3, max_q=3, m=12,\n",
    "                           start_P=0, seasonal=True,\n",
    "                           d=1, D=1, trace=True,\n",
    "                           error_action='ignore',  \n",
    "                           suppress_warnings=True, \n",
    "                           stepwise=True)\n",
    "\n",
    "print(stepwise_model.aic())\n",
    "\n",
    "stepwise_model.fit(df_sales[:-6].sales.values)\n",
    "\n",
    "stepwise_model.predict(n_periods=6)"
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