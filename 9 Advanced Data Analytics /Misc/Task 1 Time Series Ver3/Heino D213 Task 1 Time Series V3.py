# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 21:31:41 2024

@author: ntcrw
"""

#import matplotlib.dates as mdt
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as sts
import warnings

#from datetime import datetime
from math import sqrt
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

#from sympy import S, symbols, printing

warnings.filterwarnings('ignore')

#Function Definitions Begin Here. #############################################

def calc_dickey_fuller(rev_array : np.array) -> tuple:
    
    """ Method to calculate Dickey_Fuller values
    
    Parameters:
        rev_array (np.array): Array of revenue values.
     
    Returns:
        results_adfuller (tuple): Tuple with the results of the ADF 
        calculation.
        
    """
    
    results_adfuller = sts.adfuller(rev_array)

    return results_adfuller

###############################################################################


def print_results_tuple(result_tuple: tuple):
    
    """ Method to print Dickey_Fuller values
    
    Parameters:
        result_tuple (tuple): Array of revenue values.
     
    Returns:
       None
        
    """
    
    print("\nADF Statistic: \t %f" % result_tuple[0])
    print("p-value: \t %f" % result_tuple[1])
    print("Critical Values: \n")
    
    # Interate through the values in the remaining parts of the tuple.
    for key, value in result_tuple[4].items():
        print("\t%s: %.3f" % (key, value))

###############################################################################

def read_series_data(file_name : str, index='Day', new_index='Date'
                     , start_date_str=None
                     , freq='D') -> pd.DataFrame():
    
    """  Create a dataframe that contains the time series data that has 
         been read from a file.
         
         Parameters:
         -----------------
         file(str):             File name of the time series.
         index(str):            The initial index of the column.
         new_index(str):        New index name for the dataframe.
         start_date_str(str):   The start date for the dataframe.
         freq(char):            Frequncy ofthe time series.
         
         
         Returns:
         -----------------
         time_series(DataFrame):  A pandas dataframe with the time series data.
    
    """
    
    # Read the data from the CSV file
    time_series_df = pd.read_csv(file_name)
    
    
    # Convert the start date from a string to a TimeStamp.
    start_date = pd.to_datetime(start_date_str)
    
    # Convert the 'Day' column the appropriate format.
    time_series_df[index] = pd.to_timedelta(time_series_df[index] - 1
                                            , unit=freq) + start_date
    
    # Rename the column to reflect more accurately reflect the contents 
    # and the format (yyyy-mm-dd). 
    time_series_df.rename(columns={'Day': 'Date'}, inplace=True)
    
    # Reset the index for the dataframe to the Date column.
    time_series_df.set_index('Date', inplace=True)
    
    
    return time_series_df

###############################################################################  
###############################################################################

def rolling_mean(days=30):
    
    """ Create the rolling mean for the data in the dataframe.
    
    Parameters
    ----------
    
    days (int):           Days back to compute
    
                                
    Retuns:
    --------
    
    None:
    
    """
    time_series_df['rolling_mean'] = time_series_df['Revenue'].rolling(window=days).mean()

###############################################################################

def rolling_std(days=30):
    
    """ Create the rolling standard deviation for the data in the dataframe.
    
    Parameters
    ----------
    
    days (int):           Days back to compute
    
                                
    Retuns:
    --------
    
    None:
    
    """
    time_series_df['rolling_std'] = time_series_df['Revenue'].rolling(window=days).std()

###############################################################################

def test_stationarity(p_value : float, critical=0.05):
    
    """ Method to test for stationarity
    
    Parameters:
        p_value (float): The resultant p-value.
     
    Returns:
       None
        
    """
    
    if p_value <= critical:
        print("\nReject the null hypothesis H0, the data is stationary.")
    else:
        print("\nAccept the null hypothesis, the data is non-stationary. ") 
###############################################################################

# Functions End Here. ##########################################################
###############################################################################

# Read in the file and create the dataframe. **********************************

time_series_df = read_series_data(file_name='medical_time_series .csv'
                                  , index='Day', freq='D'
                                  , start_date_str='2020-01-01')

# Print some information and a summary of the newly created dataframe.********
#print(time_series_df.info())

#print(time_series_df.shape)

#print(time_series_df.head(15))
#print("Sample: \n", time_series_df.sample(5, random_state=247))

# Create the rolling mean. ****************************************************
    
rolling_mean(days=5)



#create the rolling standard deviation. ****************************************
rolling_std(days=5)

#print(time_series_df.info())
#print(time_series_df.head(15))



#(GeeksforGeeks, 2022)

#Create the visualization to show the time series data.  
#The graph below shows the time series data after the day is converted to a 
#date format. The method used was to **polyfit** and **polyld** methods.  
"""
plt.figure(figsize=[30,20])

plt.rcParams.update({'font.size': 20})

# Add labels to the graph 
plt.title("WGU Hospital System Daily Revenue 2020-2022")
plt.xlabel("Date")
plt.ylabel("Daily Hospital Revenue (in millions of USD)")

# Plot the time series data.
plt.plot(time_series_df)

# Create the trendline for the data. 
# Convert datetime objects to Matplotlib dates. 
# (matplotlib.dates — Matplotlib 3.8.2 Documentation, n.d.)

x = mdt.date2num(time_series_df.index)
y = time_series_df.Revenue
z = np.polyfit(x, y, 1)
p = np.poly1d(z)

plt.legend(['Medical Time Series'])
plt.grid()

# Show the plot.
plt.show() 


# Reset the plot parameters to show the plot properly.
plt.rcdefaults()

x = pd.Series(time_series_df.index.values)
x2 = pd.Series(range(time_series_df.shape[0]))


fig, (ax1, ax2) = plt.subplots(2,1, figsize=(25 ,10), sharex=True, sharey=False)
fig.suptitle('Rolling Mean and Standard Deviation', fontsize=40)

ax1.set_title("Rolling Mean for Revenue")
ax1.set_ylabel('Revenue in Millions of Dollars' )

ax2.set_title("Rolling Standard Deviation for Revenue")
ax2.set_ylabel('Revenue in Millions of Dollars' )

ax1.plot(x, time_series_df['rolling_mean'], color='green')
ax2.plot(x, time_series_df['rolling_std'], color='blue')
"""

# C3. Evaluation of Stationarity. #############################################

# Call the function to calculate Dickey-Fuller and output the results.
results = calc_dickey_fuller(time_series_df['Revenue'].values)



# Print the results.
#print(print_results_tuple(results))


# Test for the critical value
test_stationarity(results[1])


# Apply differencing to the data.

med_stationary_df = time_series_df.diff(periods=1, axis=0).dropna()


results = calc_dickey_fuller(med_stationary_df['Revenue'].values)



# Print the results.
print_results_tuple(results)






"""
print(med_stationary_df.info())
print(med_stationary_df.head())



print("\n\nTime Series: Original")
print(time_series_df.info())
print(time_series_df.head(15))
"""

# Train, Test splitting of the data. ******************************************
train_df = time_series_df.iloc[ : -30]
test_df = time_series_df.iloc[-31 : ]

"""
print(train_df.shape)
print(train_df.info())

print(test_df.info())
print(test_df.shape)

print(train_df.describe())
"""

# Create a model using auto find of auto_arima.
results_pdq = auto_arima(train_df['Revenue'], trace=True, supress_warnings=True) 

#Create the model for the data using the parameters found by the auto_arima 
# function.

arima_model = ARIMA(time_series_df['Revenue'], order=(1,1,0))

# Fit the model.***************************************************************
model_results = arima_model.fit()


# print the model summary.
#print(model_results.summary())



# Make a 30 day forecast.******************************************************
print("\nForecast: \n\n",model_results.forecast(30))

print(time_series_df.tail(30))


# Make predictions for the last 30 days of the data.***************************
# 700 
model_predictions = model_results.predict(start=700, end=730, type='levels')

print(model_predictions)
"""

fig, ax = plt.subplots(1,1, figsize = (20, 15))
pred = plt.plot(model_predictions, "b", label='Model Predictions')
plt.plot(test_df['Revenue'], "r", label='Test Data')
plt.xlabel("Date Index")
plt.ylabel("Revenue")
title = 'Model Predictions vs Test Data'
plt.legend()
plt.grid()

ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_minor_locator(mdates.MonthLocator())
ax.xaxis.set_minor_formatter(mdates.DateFormatter('\n%b'))
plt.title(title)
 

# Plot the Diagnostics of the model

model_results.plot_diagnostics()

# Look at the root mean square error(RMSE).
print(test_df['Revenue'].mean())
rmse = sqrt(mean_squared_error(model_predictions, test_df['Revenue']))

print(rmse)
"""

model_forecasts_30 = model_results.forecast(steps=30)


get_forecasts = model_results.get_forecast(30)



y_hat = get_forecasts.predicted_mean
conf_int = get_forecasts.conf_int(alpha=0.05)

print("\n\n\n Y_YHAT CONFIDENCE",type(conf_int))
print("\nContents of the y-hat conf: ", conf_int.head(30) )



fig, ax = plt.subplots(1,1, figsize = (20, 15))
plt.rcParams.update({'font.size': 22})
predictions = plt.plot(model_predictions, "b", label='Model Predictions')

forecasts = plt.plot(model_forecasts_30, "r", label='Model Forecasts')   # new ******************


plt.fill_between(conf_int.index, conf_int['lower Revenue'], conf_int['upper Revenue'], color='gray'  )

plt.plot(test_df['Revenue'], "g", label='Test Data')
plt.xlabel("Date Index",fontsize=20)
plt.ylabel("Revenue",fontsize=20)
plt.title('Model Predictions, Test Data, And 30-Day Predictions',fontsize=30)


plt.legend()
plt.grid()

ax.set_facecolor('lightgrey')

ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_minor_locator(mdates.MonthLocator())
ax.xaxis.set_minor_formatter(mdates.DateFormatter('\n%b'))






