# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 06:08:38 2023

@author: Matthew Heino

"""
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np

import pandas as pd
import seaborn as sns
import statsmodels.api as sm


# split the data into train and test
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor


pd.set_option('display.max_columns', 20)


# Read the the reduced CSV file.
mlr_df = pd.read_csv('Heino_reduced_medical.csv')

print("Info: ", mlr_df.info())
#print("\nContents: \n", mlr_df.head(5))


# Split the set into target and the predictors.
y = mlr_df.Initial_days
X = mlr_df.iloc[:, :-1]

print(y)
print(X.head(5))


# create the model.
X = sm.add_constant(X)
mlr_model = sm.OLS(y, X)
model_results = mlr_model.fit()

# Print the results
print(model_results.summary())


#mse = mlr_model.mse_resid

#print(y)



# E. Section*******************************************************************
# Residual Plot for the reduced set.

print(model_results.resid)

#fig = plt.figure(figsize=(10,10))
#fig = sm.graphics.plot_regress_exog(model_results,'Children', fig = fig)
#print( fig[0][1])


fig, axs = plt.subplots(figsize=(30,30),nrows=3, ncols=2, sharex=False
                        , sharey=False)
fig.tight_layout(pad=0.0)


axs[0,0] =sm.graphics.plot_regress_exog(model_results,'Arthritis')
axs[1,0] =sm.graphics.plot_regress_exog(model_results,'Diabetes')

plt.show()






#split the set

"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.30
                                                    , random_state =1)

mlr_pred_mod = sm.OLS(y_train, X_train).fit()
print(mlr_pred_mod.summary())
print("\n\nThe coefficient of determination (R-squared)", mlr_pred_mod.rsquared)
print("\n\nMSE: ", mlr_pred_mod.mse_resid)

# take the square root of the MSE = residual standard error.
mse_red = mlr_pred_mod.mse_resid

rse = np.sqrt(mse_red)

print("\n\nRSE is: ", rse)
"""











