# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 13:45:49 2024

@author: ntcrw
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdt
import numpy as np
import pandas as pd
import warnings

import statsmodels.tsa.stattools as sts

#from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from scipy import signal

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf


warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# Functions Begin Here ########################################################

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

def create_spectral_plt(dataframe, column, ax, i:int, graph_title:str, legend_text:str) -> plt:
# (matplotlib.pyplot.semilogy — Matplotlib 3.8.2 Documentation, n.d.)
# (SciPy.Signal.Periodogram — SciPY V1.12.0 Manual, n.d.)

    """ Method to create a spectral density graph. 
    
    Parameters:
        dataframe (dataframe): Dataframe with the data.
        column(str): column name
        ax(): position
        i(int): index for the graph item
        graph_title(str): Title for the graph.
        legend_text:str Legend information
     
    Returns:
       None
        
    """
    
    f, Pxx = signal.periodogram(dataframe[column])
    ax[i].semilogy(f, Pxx, label=legend_text)
    ax[i].set_title(graph_title)
    ax[i].legend()
    
    return ax[i]

#########################################################################################


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






##############################################################################
def read_series_data(file_name : str, index='Day', new_index='Date', start_date_str=None
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
    
    #print("In read series, file name: ", file)
    #print("Index  value: ", index)
    #print("New Index value: ", new_index)
    #print("The start date: ", start_date_str)
    
    # Read the data from the CSV file
    time_series_df = pd.read_csv(file_name)
    
    
    # Convert the start date from a string to a TimeStamp.
    start_date = pd.to_datetime(start_date_str)
    
    #print(type(start_date))
    
    
    # Convert the 'Day' column the appropriate format.
    time_series_df[index] = pd.to_timedelta(time_series_df[index] - 1
                                            , unit=freq) + start_date
    
    # Rename the column to reflect more accurately reflect the contents 
    # and the format (yyyy-mm-dd). 
    time_series_df.rename(columns={'Day': 'Date'}, inplace=True)
    
    # Reset the index for the dataframe to the Date column.
    time_series_df.set_index('Date', inplace=True)
    
    
    #Print info
    #print(time_series_df.head())
    
    
    return time_series_df

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





def cal_rolling_mean_std(ts_df : pd.DataFrame
                         , num_days= 31) ->  pd.DataFrame():
    
    """  Calculate the rilling mean and standard dviation for the dataframe.
         
         Parameters:
         -----------------
         ts_df(Dataframe):      Time series dataframe.
         num_days(int):         Number of days in the period.
         
         Returns:
         -----------------
         time_series(DataFrame):  A pandas dataframe with the time series data.
    
    """
    
    #print("IN method: ")
    # print("Number of Days:", num_days)
    #print("Contents of the dataframe: \n", ts_df.head())
    
    # Calculate the rolling mean for the Revenue column.
    ts_df['rolling_mean'] = ts_df['Revenue'].rolling(window=num_days).mean()
    
    # Calculate the rolling standard deviation for Revenue column.
    ts_df['rolling_std'] = ts_df['Revenue'].rolling(window=num_days).mean()
    
    
   # print("Mean: \n", ts_df['rolling_mean'])
   # print("Standard deviation: \n", ts_df['rolling_std'])
   #  print("Columns:", ts_df.columns)
    
    return ts_df

###############################################################################

# Functions End Here ##########################################################
###############################################################################


# Read in the data from the CSV file. #########################################

med_time_series_df = read_series_data(file_name='medical_time_series .csv'
                                      , start_date_str= '2020-01-01')

# Print some data about the returned data frame
#print(med_time_series_df.info())
#print("\n\n")
#print(med_time_series_df.shape)
#print("\n\n")
#print(med_time_series_df.head())

# Drop the zero values from the dataframe.
#med_time_series_df = med_time_series_df[med_time_series_df['Revenue'] != 0]

#print(med_time_series_df.head())

#print(med_time_series_df.describe())
#print(med_time_series_df.info())



# Create the rolling average for the medical time series data frame
med_time_series_df = cal_rolling_mean_std(med_time_series_df, num_days=30)


#print(med_time_series_df)

# Check for missing values in the frame.
#print(med_time_series_df.isnull().any())


# Plot the visualization of the data. #########################################

"""
plt.figure(figsize=[30,20])

plt.rcParams.update({'font.size': 20})

# Add labels to the graph 
plt.title("WGU Hospital System Daily Revenue 2020-2022")
plt.xlabel("Date")
plt.ylabel("Daily Hospital Revenue (in millions of USD)")

# Plot the time series data.
plt.plot(med_time_series_df)
plt.legend()
# Create the trendline for the data. 
# Convert datetime objects to Matplotlib dates. 
# (matplotlib.dates — Matplotlib 3.8.2 Documentation, n.d.)

x = mdt.date2num(med_time_series_df.index)
y = med_time_series_df.Revenue
z = np.polyfit(x, y, 1)
p = np.poly1d(z)

#plt.plot(x, p(x), 'g--', linewidth = '3')

# Show the plot.
plt.show() 
"""


# Plot the rolling mean and the standard deviation. ###########################
"""
x = pd.Series(med_time_series_df.index.values)

#print(x)

x2 = pd.Series(range(med_time_series_df.shape[0]))
#print(x2)


fig, ax = plt.subplots(1,1, figsize=(10 ,10), sharex=True, sharey=True)

ax.plot(x, med_time_series_df['rolling_mean'], color='green')
ax.plot(x, med_time_series_df['rolling_std'], color='blue')
"""


###############################################################################
###############################################################################

# C3. Evaluation of Stationarity. #############################################

## Insert code here
# Call the function to calculate Dickey-Fuller and output the results.
results = calc_dickey_fuller(med_time_series_df['Revenue'].values)

# Print the results.
#print_results_tuple(results) 


# Test for the critical value
test_stationarity(results[1])

# Apply differencing to the data.




med_stationary_df = med_time_series_df.diff(axis=0).dropna()

#print(med_stationary_df.info())
#print(med_stationary_df.shape)


results = calc_dickey_fuller(med_stationary_df['Revenue'].values)

# Print the results.
#print_results_tuple(results)

# Test for the critical value
#test_stationarity(results[1])


# Plot the transformed data.

"""
#plt.legend()
#med_stationary_df['Revenue'].plot(figsize=[20,7], legend=['Revenue']
#                                  , title="Differenced  Revenue Data")

#med_stationary_df['rolling_mean'].plot(figsize=[20,7], legend=['Mean']
#                                  , title="Differenced  Revenue Data")


"""
# Ouput the stationary dataset to a CSV file for submission.

#med_stationary_df.to_csv('Heino D213 Task Stationary.csv')


# Create the train and test set.
# Do not shuffle to keep the series intact.
train, test = train_test_split(med_stationary_df, test_size=0.2, train_size=.80
                               , shuffle=False, random_state=247)

#$print(test.info())
#print(train.info())

#print(train)


seasonal_decomp = seasonal_decompose(med_stationary_df['Revenue'])
#print(seasonal_decomp.seasonal)

#Plot the data.
"""
# Show a graph of the seasonal decomposed data. 
plt.figure(figsize=[30, 10])
plt.rcParams.update({'font.size': 25})
plt.xlabel("Year and Month")
plt.title("Seasonality From 2020-2022")

# Plot the seasonal component of the data.
plt.plot(seasonal_decomp.seasonal)

"""
#############################################################################

"""
plt.figure(figsize=[20, 7])
plt.rcParams.update({'font.size': 18})

plt.xlabel("Year, Month, and Day")
plt.title("Seasonality From January 31, 2020 to Mar 1, 2020")
plt.ylim(-0.08, 0.06)
plt.xlim(pd.to_datetime('2020-01-31'), pd.to_datetime('2020-03-30'))
plt.grid(linestyle='--')

plt.plot(seasonal_decomp.seasonal, marker='o')

plt.axhline(y=0, linewidth=2, linestyle=':',color='r')
plt.axvline(x=pd.to_datetime('2020-02-08'), color='darkgreen', linestyle='dashdot')
plt.axvline(x=pd.to_datetime('2020-02-15'), color='darkgreen', linestyle='dashdot')
plt.axvline(x=pd.to_datetime('2020-02-22'), color='darkgreen', linestyle='dashdot')
plt.axvline(x=pd.to_datetime('2020-02-29'), color='darkgreen', linestyle='dashdot')
plt.axvline(x=pd.to_datetime('2020-03-07'), color='darkgreen', linestyle='dashdot')
plt.axvline(x=pd.to_datetime('2020-03-14'), color='darkgreen', linestyle='dashdot')
plt.axvline(x=pd.to_datetime('2020-03-21'), color='darkgreen', linestyle='dashdot')
plt.axvline(x=pd.to_datetime('2020-03-28'), color='darkgreen', linestyle='dashdot')

"""

# 3. Check for trends. ########################################################
"""
plt.rcParams.update({'font.size': 25})
plt.figure(figsize=[30, 9])
plt.xlabel("Year and Month")
plt.title("Trends From January 1,2020 to December 31, 2021")

plt.axhline(y=0, linewidth=2, linestyle=':',color='r')
plt.plot(seasonal_decomp.trend, marker='.')
plt.axvline(x=pd.to_datetime('2021-06-13'), color='darkgreen', linestyle='dashdot')
plt.grid()

"""

#  4) The autocorrelation function.
# Plot the auto correlation

"""
lag = 30

fig, (ax,ax2) = plt.subplots(2,1, figsize=[20,15], sharex=True)
fig.tight_layout(pad=3.0)


# Plot the ACF graph.
plot_acf(med_stationary_df['Revenue'], lags=lag, zero=False
         , title="Autocorrelation Using Plot_ACF", ax=ax)
ax.set_ylim((-.20, 0.50))
ax.set_xlabel("Lags")
ax.set_facecolor("lightGray")
ax.grid()

#Plot the PACF graph.
plot_pacf(med_stationary_df['Revenue'], lags=lag, zero=False
         , title="Partial Autocorrelation Using Plot_PACF", ax=ax2)
ax2.set_xlabel("Lags")
ax2.set_facecolor("lightGray")
ax2.set_ylim((-.20, 0.50))
ax2.grid()



# Plot both on the same graph for comparison.#################################

# (statsmodels.tsa.stattools.pacf - Statsmodels 0.15.0 (+200), n.d.)
# (statsmodels.tsa.stattools.acf - Statsmodels 0.14.1, n.d.)

revenue_acf_df = acf(med_stationary_df['Revenue'], nlags=lag, missing='drop')

# Will use the default Yule-Walker. 
revenue_pacf_df = pacf(med_stationary_df['Revenue'], nlags=lag) 


# Create a pandas dataframe to hold the data from the correlation.
acf_pacf_df = pd.DataFrame([revenue_acf_df, revenue_pacf_df]).T

# Set the columns
acf_pacf_df.columns = ['ACF','PACF']
acf_pacf_df.drop(index=0, inplace=True)

fig, ax = plt.subplots(1,1, figsize=[20, 10], sharex=True)
fig.tight_layout(pad=3.0)

ax = acf_pacf_df.plot(kind='bar', color=('green', 'blue'), title="ACF and PCAF", ax=ax)
ax.set_facecolor("lightGray")
ax.grid()
 
"""

# 5. Spectral Density. #######################################################
"""
# Create a spectral density plot. Using a periodogram.
fig,ax = plt.subplots(2,1, figsize=(20,20), sharex=True, sharey=True)
fig.suptitle('Spectral Density for Original Data and the Stationary Data', fontsize=30)

# Create a plot for the first set of data the initial dataframe.
create_spectral_plt(dataframe=med_time_series_df,ax=ax, column='Revenue'
                    ,i=0, graph_title='Original Data', legend_text='Original Data Spectral')

create_spectral_plt(dataframe=med_stationary_df,ax=ax, column='Revenue'
                    ,i=1, graph_title='Stationary Data', legend_text='Stationary Data Spectral')



# Using the matplot version to view the spectral density graph of the 
# stationary data.
fig,ax = plt.subplots(1,1, figsize=(30,10))
fig.suptitle('Spectral Density Using PSD', fontsize=40)

psd = plt.psd(x=med_stationary_df.Revenue)
"""

# 6. The Decompsed time series Data. ##########################################

# Create a graph of the deompsed seasonal data.
# The decomposition of the time series data.
# Using plot() function to plot the data.

# Reset the plot parameters to show the plot properly.
"""
plt.rcdefaults()

newplot = seasonal_decomp.plot()
newplot.set_figwidth(15)
newplot.set_figheight(20) 
newplot.suptitle('The Decomposed Time Series', fontsize=30, y=1.0 )
"""


# 7. LAck of Trends in the Residuals.##########################################
##Identification of ARIMA model.

"""
# The decomposition of the time series data. 

fig,ax = plt.subplots(1,1, figsize=(35,10))
fig.suptitle('Residuals of the Decomposition (Seasonal Data)', fontsize=50)

plt.axhline(y=0, linewidth=2, linestyle=':',color='r')
plt.plot(seasonal_decomp.resid) 
"""



pmd_auto_arima = auto_arima(train['Revenue'], trace=True)

# print the summary for auto arima.
print(pmd_auto_arima.summary())


# Forecasting using the standard ARIMA model.
arima_model = ARIMA(train['Revenue'], order=(0, 0, 2))
arima_fitted = arima_model.fit()


arima_fitted.summary()


# (Statsmodels.Regression.Linear_Model.OLSResults.Get_Prediction - Statsmodels 0.15.0 (+200), n.d.)


col= {'predicted_mean' : 'Revenue'}

# Will return a prediction and the prediction variance.
forecasts = arima_fitted.get_prediction(start=560, end=700, dynamic=True)

forecasts_df = pd.DataFrame(forecasts.predicted_mean)

# Rename the predicted_mean column to represent what is stored there. 
# The differences in revenue experienced on a daily basis.

forecasts_df.rename(columns=col, inplace=True)

# print contents.
print(forecasts_df.head())

print("\n Sample: \n", forecasts_df.sample(10))



# Create a dataframe that has the predictions and the test data.
train_forecast_df = pd.concat([train.copy(deep=True)
                               , forecasts_df.copy(deep=True)])

train_forecast_df = train_forecast_df.cumsum()



print(train_forecast_df.sample(5))
print(train_forecast_df.head())


print(train_forecast_df)


confidence_int = forecasts.conf_int()



#print("\n Confidence Interval: ",confidence_int)



august_row_df = pd.DataFrame({'lower Revenue':[18.372498 ]
                              , 'upper Revenue': [18.372498], 'Date': '2021-08-12' } )

august_row_df['Date'] = pd.to_datetime(august_row_df.Date)
august_row_df.set_index('Date', inplace=True)

#print(august_row_df)

#print(august_row_df.info())


confidence_int = pd.concat([august_row_df, confidence_int])
#print(confidence_int)

# Transform the columns back into actual revenue.
confidence_int = confidence_int.cumsum()

# Remove the first row by taking the interval not including the first row 
# in the new dataframe.

confidence_int = confidence_int.loc['2021-08-13' : '2021-12-31']

#print(confidence_int)


# Create a visual of the predictions.


train_revenue = train.copy(deep=True)
test_revenue = test.copy(deep=True)
revenue_df = pd.concat([train_revenue, test_revenue])



revenue_df = revenue_df.cumsum()

#print(type(train_revenue))
#print(type(train_revenue))


# Reconstitute the data. 



#print(revenue_df.sample(50))


"""
fig, ax = plt.subplots(1,1, figsize = [20, 5])

ax.set_facecolor('lightgray')
plt.title("WGU Hospital System Revenue From January 2020 to January 2022")
plt.xlabel('Date')
plt.ylabel("Daily Revenue (millions USD)")

# Change the limit on the y-axis to make it dispaly more appropriately.
plt.ylim(-5, 30)
plt.grid()

# plot the forecasted datae
plt.plot(train_forecast_df['Revenue'], color='darkgreen',linestyle='dashed', linewidth=2)

# plot the original data.
#plt.plot(train_revenue['Revenue'], color='blue')
plt.plot(revenue_df['Revenue'], color='blue')

#Plot the confidence interval 
plt.fill_between(confidence_int.index, confidence_int['lower Revenue']
                 , confidence_int['upper Revenue'], color='pink')

plt.legend(['Actual', 'Predicted'])

plt.show()

"""
"""
print(train.tail())
#print(test.info())



# Retrieve the date for reference
# Needed the data for later tasks.

#row_date = med_time_series_df.iloc[-141]
#
row_date = train.iloc[-1]
row_date2 = test.iloc[:1]

print("\n\nRow Data: ", row_date)
print("\n\nRow Data: ", row_date)row_date = train.iloc[-1]

row_date2 = test.iloc[:1]
print("\n\nRow Data: ", row_date)
print("\n\nRow Data: ", row_date)

print("\n\n Date: ",med_time_series_df.loc['2021-08-12'])

# 2021-08-12 ??????????????????????????????????????????????????????????????????
#  18.372498 
"""
"""
plt.rcdefaults()
fig, ax = plt.subplots(1,1, figsize = [20, 5])
ax = arima_fitted.plot_diagnostics()
"""
"""
print("\nTrain data: \n",train_forecast_df['2021-08-12':'2021-12-31']['Revenue'])


print("Train lenght: ", len(train_forecast_df['2021-08-12':'2021-12-31']['Revenue']))

print("Total Revenue: ", revenue_df['2021-08-12':'2021-12-31']['Revenue'])
print("Total Revenue: ", len(revenue_df['2021-08-12':'2021-12-31']['Revenue']))

"""
#root_mse = mean_squared_error(revenue_df['2021-12-31':'2021-12-31']
#                              ,train_forecast_df['2021-08-12':'2021-12-31'])

test1 =  train_forecast_df['2021-08-12':'2021-12-31']['Revenue'].to_numpy()
test2 =  revenue_df['2021-08-12':'2021-12-31']['Revenue'].to_numpy()
print(test1)
print(test2)


root_mse = mean_squared_error(test1, test2, squared=False)
mse = mean_squared_error(test1, test2, squared=True)

print(root_mse)
print(mse)












