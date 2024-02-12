# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 10:12:43 2024

@author: Matthew Heino

Purpose:
    Task 1 File for time series asssessment.
    
    
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdt
import numpy as np
import pandas as pd
import warnings

import statsmodels.tsa.stattools as ts

#from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from scipy import signal
from sklearn.model_selection import train_test_split

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf

# Pre-assessment Tasks #######################################################
#
# 1) Read in data from the CSV file 
# 2) Retrieve info about the data in the file.
# 3)
#
###############################################################################

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# Read in the data from the CSV file.
med_time_series_df = pd.read_csv('medical_time_series .csv')

# View some information about te data in the dataframe.
med_time_series_df.info()

#View the data in the dataframe.
#print(med_time_series_df.head())
#print(med_time_series_df.tail())


# Functions ####################################################################

def calc_dickey_fuller(rev_array : np.array) -> tuple:
    
    """ Method to calculate Dickey_Fuller values
    
    Parameters:
        rev_array (np.array): Array of revenue values.
     
    Returns:
        results_adfuller (tuple): Tuple with the results of the ADF 
        calculation.
        
    """
    
    results_adfuller = ts.adfuller(rev_array)

    return results_adfuller
    
def print_results_tuple(result_tuple: tuple):
    
    """ Method to print Dickey_Fuller values
    
    Parameters:
        result_tuple (tuple): Array of revenue values.
     
    Returns:
       None
        
    """
    
    print("ADF Statistic: %f" % result_tuple[0])
    print("p-value: %f" % result_tuple[1])
    print("Critical Values: ")
    
    # Interate through the values in the remaining parts of the tuple.
    for key, value in result_tuple[4].items():
       # print("IN FOR")
        print("\t%s: %.3f" % (key, value))
        
def create_spectral_plt(dataframe, column, ax, i:int, graph_title:str) -> plt:
# (matplotlib.pyplot.semilogy — Matplotlib 3.8.2 Documentation, n.d.)
# (SciPy.Signal.Periodogram — SciPY V1.12.0 Manual, n.d.)

    """ Method to create a spectral density graph. 
    
    Parameters:
        dataframe (dataframe): Dataframe with the data.
        column(str): column name
        ax(): position
        i(int): index for the graph item
        graph_title(str): Title for the graph.
     
    Returns:
       None
        
    """
    #print("Vlaue if ax:", ax)
   #print("In function")
    
    f, Pxx = signal.periodogram(dataframe[column])
    ax[i].semilogy(f, Pxx, label='Spectral Density')
    ax[i].set_title(graph_title)
    ax[i].legend()
    
    return ax[i]
    
    

def test_stationarity(p_value : float, critical=0.05):
    
    """ Method to test for stationarity
    
    Parameters:
        p_value (float): The resultant p-value.
     
    Returns:
       None
        
    """
    
    if p_value <= critical:
        print("Reject the null hypothesis H0, the data is stationary.")
    else:
        print("Accept the null hypothesis, the data is non-stationary. ")


# Section C. ##################################################################
#
#  Tasks:
#   1)   Change the data in the from an index of number values to a date 
#        index that can be used for plotting time-based data.
#
#   2)  
#
###############################################################################


#C1. Time Series Visualization ################################################
#
#  Task:
#    Change the numeric index into a date that can be used to visualize the 
#    data.  Will make use of the pnadas to_datetime mehtod.
#
#    Change the date column to be the new index for the dataframe.
#
###############################################################################

# Start date initialization
start_date = pd.to_datetime('2020-01-01')

#print(start_date)
#print(type(start_date))

# Convert the Day column to a date using pandas
# (Pandas.to_Timedelta — Pandas 2.2.0 Documentation, n.d.)

med_time_series_df['Day'] = pd.to_timedelta(med_time_series_df['Day'] - 1
                                            , unit='D') + start_date

#med_time_series_df.columns = ['Date', 'Revenue']

med_time_series_df.rename(columns={"Day": "Date"}, inplace=True)

med_time_series_df.set_index('Date', inplace=True)

#print(med_time_series_df.columns)
#print(med_time_series_df.index)


# Create a visualization of the time series data #############################
"""
plt.figure(figsize=[30,20])

plt.rcParams.update({'font.size': 20})

# Add labels to the graph 
plt.title("WGU Hospital System Daily Revenue 2020-2021")
plt.xlabel("Date")
plt.ylabel("Daily Hospital Revenue (in millions of USD)")


# Plot the time series data.
plt.plot(med_time_series_df) 

# Create the trendline for the data. 
# Convert datetime objects to Matplotlib dates. (matplotlib.dates — Matplotlib 3.8.2 Documentation, n.d.)

x = mdt.date2num(med_time_series_df.index)
y = med_time_series_df.Revenue
z = np.polyfit(x, y, 1)
p = np.poly1d(z)

plt.plot(x, p(x), 'g-', linewidth = '5' )

# Show the plot.
plt.show()        

"""

# C2. ########################################################################



##############################################################################


# C3. Stationarity of the Data. ##############################################
# Task:
#   Check the stationarity of the dataset. By performing a an Augmented 
#   Dickey-Fuller test   
#    
###############################################################################
 
# Perfrom the adfuller test.

# Call the function to calculate DickeyFuller and output the results.
results = calc_dickey_fuller(med_time_series_df['Revenue'].values)


# Print the results.
#print_results_tuple(results)

# Test for the critical value
#test_stationarity(results[1])


# apply the diff() method to the data.

med_stationary_df = med_time_series_df.diff().dropna()

#print(med_stationary_df.info())
#print(med_stationary_df.head())


# Perform ADF on the "differenced" data.
results_stationary = calc_dickey_fuller(med_stationary_df['Revenue'].values)

print_results_tuple(results_stationary)

# Test for the critical value
test_stationarity(results_stationary[1])

# Plot the transformed data.
#med_stationary_df.plot()


#test = med_time_series_df[med_time_series_df['Revenue'] >= -0.292356]

#print(test)

# C5. Copy of the Prepared Data set.###########################################

# Ouput the stationary dataset to a CSV file for submission.

#med_stationary_df.to_csv('Heino D213 Task Stationary.csv')

###############################################################################


###############################################################################
# D. Model Indentification and Analysis. ######################################
# Tasks:
#    
#  1) Split the data into training and test set for the model.
#  2) Check for seasonality.
#  3) Check for trends.
#  4) The autocorrelation function.
#  5) The decomposed time series. 
#  6) Confirmation of the lack of trends in the residuals of the decomposed 
#     time series
#    
##############################################################################

# Splitting the data into training and test set using train_test_split
# Do not shuffle to keep the series intact.
train, test = train_test_split(med_stationary_df, test_size=0.2
                               , train_size=0.80, shuffle=False
                               , random_state=247) 


# print the contents of the train and test sets.
#print("\n\n")
#print(train)
#print(len(train))

#print("\n\n")
#print(test)
#print(len(test))

##############################################################################


# 2) Check for seasonality. ###################################################


seasonal_decomp = seasonal_decompose(med_stationary_df)
#print(type(seasonal_decomp))

"""
# Show a graph of the seasonal decomposed data. 

plt.figure(figsize=[20, 7])
plt.rcParams.update({'font.size': 17})
plt.xlabel("Month and Year")
plt.title("Seasonality From 2020-2021")

# Plot the seasonal component of the data.
plt.plot(seasonal_decomp.seasonal)


plt.figure(figsize=[20, 7])
plt.xlabel("Month and Year")
plt.title("Seasonality From January 1,2020 to February 1. 2021")
plt.plot(seasonal_decomp.seasonal, marker='o')


plt.xlim(pd.to_datetime('2020-01-01'), pd.to_datetime('2020-02-01'))
plt.axvline(x=pd.to_datetime('2020-01-06'), color='darkgreen', linestyle='dashdot')
plt.axvline(x=pd.to_datetime('2020-01-13'), color='darkgreen', linestyle='dashdot')
plt.axvline(x=pd.to_datetime('2020-01-20'), color='darkgreen', linestyle='dashdot')
plt.axvline(x=pd.to_datetime('2020-01-27'), color='darkgreen', linestyle='dashdot')

"""

##############################################################################

# 3) Check for trends.
# Using a plot from matplotlib


"""
plt.figure(figsize=[20, 7])
plt.xlabel("Year and Month")
plt.title("Seasonality From January 1,2020 to February 1. 2021")
plt.plot(seasonal_decomp.trend, marker='.')
"""

##############################################################################

#  4) The autocorrelation function.
# Plot the auto correlation
# Use 8 lags to correspond to the seven days in a week.

lag = 10

"""
fig, (ax,ax2) = plt.subplots(2,1, figsize=[10,10], sharex=True)

plot_acf(med_stationary_df, lags=lag, zero=False
         , title="Autocorrelation Using Plot_ACF", ax=ax)
ax.set_xlabel("Lags")



plot_pacf(med_stationary_df, lags=lag, zero=False
         , title="Partial Autocorrelation Using Plot_PACF", ax=ax2)
ax2.set_xlabel("Lags")


"""


#(statsmodels.tsa.stattools.pacf - Statsmodels 0.15.0 (+200), n.d.)
revenue_acf_df = acf(med_stationary_df, nlags=lag, missing='drop')


# Will use the default Yule-Walker. 
revenue_pacf_df = pacf(med_stationary_df, nlags=lag) 

print(revenue_acf_df)
print(revenue_pacf_df)

# Create a pandas dataframe to hold the data from the correlation.


acf_pacf_df = pd.DataFrame([revenue_acf_df, revenue_pacf_df]).T

# Set the columns
acf_pacf_df.columns = ['ACF','PACF']
acf_pacf_df.drop(index=0, inplace=True)
#acf_pacf_df.index += 1

print(acf_pacf_df.info())
"""
# plot the points.

acf_pacf_df.plot(kind='bar', color=('green', 'blue'), title="ACF and PCAF")

"""

###############################################################################
# Create a spectral density plot. Using a periodogram.
"""

fig,ax = plt.subplots(2,1, figsize=(20,10), sharex=True
                                  , sharey=True)
fig.suptitle(' Spectral Density for Original Data and the Stationary Data'
             , fontsize=30)
# Create a plot for the first set of data the initial dataframe.

create_spectral_plt(dataframe=med_time_series_df,ax=ax, column='Revenue'
                    ,i=0, graph_title='Original Data')

create_spectral_plt(dataframe=med_stationary_df,ax=ax, column='Revenue'
                    ,i=1, graph_title='Stationary Data')

"""


# Using the matplot vesion to view the spectral density graph of the 
# stationary data.


#psd = plt.psd(x=med_stationary_df.Revenue)

#print(psd)


# The decomposition of the time series data. 
#plt.plot(seasonal_decomp.resid) 



# Create a graph of the deompsed seasonal data.
# Using plot() functin to plot the data.

#seasonal_decomp.plot()
#print(seasonal_decomp)

###############################################################################

"""  
# DO NOT DELETE ###############################################################

test = med_time_series_df.index.name == '2020-01-01'

#print(test)
test = med_time_series_df.loc['2021-08-07']

print(test)



#print(test)
test = med_time_series_df.loc['2021-08-07']

print(test)


#print(test)
test2 = med_time_series_df.iloc[-146]

print(test)

###############################################################################
"""

## D2. Identification of ARIMA model.##########################################
"""
pmd_auto_arima = auto_arima(train['Revenue'], start_p=0, d=1, trace=True
                            , max_p=5, max_d=5, max_q=5, start_P=0, D=1
                            ,start_Q=0, max_P=5, max_D=5, max_Q=5, n_fits=50) 
"""

#pmd_auto_arima = auto_arima(train['Revenue'], trace=True)


# print the summary
#print(pmd_auto_arima.summary())


#model = ARIMA(train, order=(1, 0, 0), freq='D')
#results = model.fit()

#print(results.summary())

###############################################################################


# D3. Forecasting using the ARIMA model.#######################################
# (Statsmodels.Regression.Linear_Model.OLSResults.Get_Prediction - Statsmodels 0.15.0 (+200), n.d.)
# change order= 0,0,2

arima_model = ARIMA(train['Revenue'], order=(1, 0, 0), freq='D')
arima_fitted = arima_model.fit()
#print(arima_fitted.summary())


# Will return a prediction and the prediction variance.
forecasts = arima_fitted.get_prediction(start=584, end=729, dynamic=True)

##plt.plot(forecasts.prediction_results)
#plt.plot(forecasts.predicted_mean)

##print(forecasts.predicted_mean)

forecasts_df = pd.DataFrame(forecasts.predicted_mean)

# Rename the predicted_mean column to represent what is sotred there. 
# The differences in revenue experienced on a daily basis.

col= {'predicted_mean' : 'Revenue'}
forecasts_df.rename(columns=col, inplace=True)


#print(forecasts_df)

#test2 = med_time_series_df.iloc[-147]
#print(test2)

#print(test)
#test = med_time_series_df.loc['2021-08-07']

#print(test)

# Create a dataframe that has the predictions and the test data.
train_forecast_df = pd.concat([train.copy(deep=True)
                               , forecasts_df.copy(deep=True)])

train_forecast_df = train_forecast_df.cumsum()


#print(train_forecast_df)

#print(type(forecasts))

###############################################################################
# Retrieve the confidence intervals fromforecasts (PredictionResultsWrapper)
# Using the conf_int() method that is available via forecasts. 

confidence_int = forecasts.conf_int()
#print(confidence_int)

###############################################################################
# Using the confidence interval and the value found in a previous cell [] 
# Create a new dataframe to be used in conjunction with the confidence 
# interval.

august_row_df = pd.DataFrame({'lower Revenue':[19.312734]
                              , 'upper Revenue': [19.312734], 'Date': '2021-08-07' } ) 

# Convert the date string to a datetime object.

# Set this converted date string to be the index for the dataframe
august_row_df['Date'] = pd.to_datetime(august_row_df.Date)
august_row_df.set_index('Date', inplace=True)


#print("\n\n",august_row_df)
#print("\n\n")
#print("\n\n",august_row_df.info())

##################################

confidence_int = pd.concat([august_row_df, confidence_int])

# Transform the columns back into actual revenue.
confidence_int = confidence_int.cumsum()

# Remove the first row by taking the interval not including the first row 
# in the new dataframe.

confidence_int = confidence_int.loc['2021-08-08' : '2021-12-31']


#print(confidence_int)

###############################################################################

### Create a visualization that shows the predition versus what was actaully 
# recorded. 
"""
plt.figure()
plt.title("WGU Hospital System Revenue")
plt.xlabel('Date')
plt.ylabel("Daily Revenue (millions USD)")

# plot the forecasted datae
plt.plot(train_forecast_df, color='green',linestyle='dashed')

# plot the original data.
plt.plot(med_time_series_df, color='blue')

#Plot the confidence interval 
plt.fill_between(confidence_int.index, confidence_int['lower Revenue']
                 , confidence_int['upper Revenue'], color='lightgray')

# Change the limit on the y-axis to make it dispaly more appropriately.
plt.ylim(-5, 25)
plt.grid()
plt.set_facecolor()


plt.show()

"""
#

fig, ax = plt.subplots(1,1, figsize = [16, 15])

ax.set_facecolor('lightgray')
plt.title("WGU Hospital System Revenue")
plt.xlabel('Date')
plt.ylabel("Daily Revenue (millions USD)")

# plot the forecasted datae
plt.plot(train_forecast_df, color='green',linestyle='dashed')

# plot the original data.
plt.plot(med_time_series_df, color='blue')

#Plot the confidence interval 
plt.fill_between(confidence_int.index, confidence_int['lower Revenue']
                 , confidence_int['upper Revenue'], color='teal')

# Change the limit on the y-axis to make it dispaly more appropriately.
plt.ylim(-5, 25)
plt.grid()
plt.legend(['Actual', 'Predicted'])

plt.show()

###############################################################################






















