# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import pmdarima as pm
from pmdarima.model_selection import train_test_split
from pmdarima import auto_arima 

import statsmodels

# Read the AirPassengers dataset
soya_prices = pd.read_csv("D://FYP/Datasets/Linear Regression_ Indore_monthly_avg_2.csv", index_col ='Date', parse_dates = True)
  
# Print the first five rows of the dataset
soya_prices.head()

# ETS Decomposition
result = seasonal_decompose(soya_prices, model ='multiplicative')
  
# ETS plot
result.plot()



# Tests

# Dickey Fuller Test

statsmodels.tsa.stattools.adfuller(x, maxlag=None, regression='c', autolag='AIC', store=False, regresults=False)

# ACF plot

statsmodels.graphics.tsaplots.plot_acf(x, ax=None, lags=None, *, alpha=0.05, use_vlines=True, adjusted=False, fft=False, missing='none', title='Autocorrelation', zero=True, auto_ylims=False, bartlett_confint=True, vlines_kwargs=None, **kwargs)

# PACF plot

statsmodels.tsa.stattools.pacf(x, nlags=None, method='ywadjusted', alpha=None)


#2

# Ignore harmless warnings 
import warnings 
warnings.filterwarnings("ignore") 
  
# Fit auto_arima function to AirPassengers dataset 
stepwise_fit = auto_arima(airline['# Passengers'], start_p = 1, start_q = 1, 
                          max_p = 3, max_q = 3, m = 12, 
                          start_P = 0, seasonal = True, 
                          d = None, D = 1, trace = True, 
                          error_action ='ignore',   # we don't want to know if an order does not work 
                          suppress_warnings = True,  # we don't want convergence warnings 
                          stepwise = True)           # set to stepwise 
  
# To print the summary 
stepwise_fit.summary()

#3

# Split data into train / test sets 
train = airline.iloc[:len(airline)-12] 
test = airline.iloc[len(airline)-12:] # set one year(12 months) for testing 

# Fit a SARIMAX(0, 1, 1)x(2, 1, 1, 12) on the training set 
from statsmodels.tsa.statespace.sarimax import SARIMAX 

model = SARIMAX(train['# Passengers'], 
				order = (0, 1, 1), 
				seasonal_order =(2, 1, 1, 12)) 

result = model.fit() 
result.summary() 


#4

start = len(train) 
end = len(train) + len(test) - 1

# Predictions for one-year against the test set 
predictions = result.predict(start, end, 
							typ = 'levels').rename("Predictions") 

# plot predictions and actual values 
predictions.plot(legend = True) 
test['# Passengers'].plot(legend = True) 


#4

# Load specific evaluation tools 
from sklearn.metrics import mean_squared_error 
from statsmodels.tools.eval_measures import rmse 

# Calculate root mean squared error 
rmse(test["# Passengers"], predictions) 

# Calculate mean squared error 
mean_squared_error(test["# Passengers"], predictions) 

#5


# Train the model on the full dataset 
model = model = SARIMAX(airline['# Passengers'], 
						order = (0, 1, 1), 
						seasonal_order =(2, 1, 1, 12)) 
result = model.fit() 

# Forecast for the next 3 years 
forecast = result.predict(start = len(airline), 
						end = (len(airline)-1) + 3 * 12, 
						typ = 'levels').rename('Forecast') 

# Plot the forecast values 
airline['# Passengers'].plot(figsize = (12, 5), legend = True) 
forecast.plot(legend = True) 


