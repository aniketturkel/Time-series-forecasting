# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
  
# Tests

# Hello

# Read the AirPassengers dataset
soya_prices = pd.read_csv("D://FYP/Datasets/Linear Regression_ Indore_monthly_avg_2.csv",
                       index_col ='Date',
                       parse_dates = True)
  
# Print the first five rows of the dataset
#soya_prices.head()

# ETS Decomposition
result = seasonal_decompose(soya_prices, 
                            model ='multiplicative')
  
# ETS plot
result.plot()
