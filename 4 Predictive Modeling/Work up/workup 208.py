# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 20:50:18 2023

@author: ntcrw


Citation:
    
    
https://mkatzenbach8.medium.com/introduction-to-ordinary-least-squares-ols-using-statsmodels-3329d120eadd



"""

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd
"""
spector_data = sm.datasets.spector.load()
spector_data.exog = sm.add_constant(spector_data.exog, prepend=False)

# Fit and summarize OLS model
mod = sm.OLS(spector_data.endog, spector_data.exog)
res = mod.fit()
print(res.summary())
"""


x = np.array([1, 1, 2, 3, 4, 5, 5, 5, 6, 7, 10, 13, 14], dtype = np.float64)
y = np.array([4, 5, 3, 4, 7, 5, 6, 7, 9, 10, 12, 14, 18], dtype = np.float64)


d = {'X': x, 'Y': y}
df = pd.DataFrame(d)

plt.scatter(x,y)

# formula and model
f = 'y ~ x'
model = ols(f, data = df).fit()

#summary
print(model.summary())