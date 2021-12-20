# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 16:26:46 2015

@author: jrose01
"""

#from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LassoLarsCV
 
#Load the dataset
data = pd.read_csv("sample_report.csv")

#upper-case all DataFrame column names
data.columns = map(str.upper, data.columns)

#select predictor variables and target variable as separate data sets  
predvar= data[['NUM_UNITS','EQUIP_FAIL','TRAINEE','PROD_STEPS','SLEEP_HRS',
'SHIFT_HRS']]

target = data.MANUF_LEAD

# split data for scatterplots
train,test=train_test_split(data, test_size=.4, random_state=123)
 
# better variable names and labels for plots 
train['Manufacturing Lead Time']=train['MANUF_LEAD']
train['# Ingredient Units in Stock']=train['NUM_UNITS']
train['# Production Steps']=train['PROD_STEPS']
train['# Hours of Sleep']=train['SLEEP_HRS']
train['# Shift Hours Worked']=train['SHIFT_HRS']
recode1 = {1:str('Yes'), 0:str('No')}
train['Equipment Failure']= train['EQUIP_FAIL'].map(recode1)
train['Trainees Working']= train['TRAINEE'].map(recode1)

#scatterplot matrix for quantitative variables
fig1 = sns.PairGrid(train, y_vars=["Manufacturing Lead Time"], 
                 x_vars=['# Ingredient Units in Stock',\
                 '# Production Steps','# Hours of Sleep','# Shift Hours Worked'],\
                 palette="GnBu_d")
fig1.map(plt.scatter, s=50, edgecolor="white")
plt.title('Figure 1. Association Between Quantitative Predictors and Manufacturing Lead Time', 
                    fontsize = 12, loc='right')
fig1.savefig('reportfig1.jpg')
                    
# boxplots for association between binary predictors & response
box1 = sns.boxplot(x="Equipment Failure", y="Manufacturing Lead Time", data=train)
plt.title('Figure 2. Association Between Equipment Failure and Manufacturing Lead Time', 
                  fontsize = 12, loc='right')
box1 = plt.gcf()
box1.savefig('reportfig2.jpg')

box2 = sns.boxplot(x="Trainees Working", y="Manufacturing Lead Time", data=train)
plt.title('Figure 3. Association Between Trainee Involvement in Production\n and Manufacturing Lead Time', 
                  fontsize = 12, multialignment='center')
box2 = plt.gcf()
box2.savefig('reportfig3.jpg')

# standardize predictors to have mean=0 and sd=1 for lasso regression
predictors=predvar.copy()
from sklearn import preprocessing
predictors['NUM_UNITS']=preprocessing.scale(predictors['NUM_UNITS'].astype('float64'))
predictors['EQUIP_FAIL']=preprocessing.scale(predictors['EQUIP_FAIL'].astype('float64'))
predictors['TRAINEE']=preprocessing.scale(predictors['TRAINEE'].astype('float64'))
predictors['PROD_STEPS']=preprocessing.scale(predictors['PROD_STEPS'].astype('float64'))
predictors['SLEEP_HRS']=preprocessing.scale(predictors['SLEEP_HRS'].astype('float64'))
predictors['SHIFT_HRS']=preprocessing.scale(predictors['SHIFT_HRS'].astype('float64'))

# split data into train and test sets
pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, target, 
                                                              test_size=.4, random_state=123)
# specify the lasso regression model
model=LassoLarsCV(cv=10, precompute=False).fit(pred_train,tar_train)

# print variable names and regression coefficients
dict(zip(predictors.columns, model.coef_))

# plot coefficient progression
m_log_alphas = -np.log10(model.alphas_)
ax = plt.gca()
plt.plot(m_log_alphas, model.coef_path_.T)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.ylabel('Regression Coefficients')
plt.xlabel('-log(alpha)')
plt.title('Regression Coefficients Progression for Lasso Paths')

# plot mean square error for each fold
m_log_alphascv = -np.log10(model.cv_alphas_)
plt.figure()
plt.plot(m_log_alphascv, model.cv_mse_path_, ':')
plt.plot(m_log_alphascv, model.cv_mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.legend()
plt.xlabel('-log(alpha)')
plt.ylabel('Mean squared error')
plt.title('Mean squared error on each fold')
         
# MSE from training and test data
from sklearn.metrics import mean_squared_error
train_error = mean_squared_error(tar_train, model.predict(pred_train))
test_error = mean_squared_error(tar_test, model.predict(pred_test))
print ('training data MSE')
print(train_error)
print ('test data MSE')
print(test_error)

# R-square from training and test data
rsquared_train=model.score(pred_train,tar_train)
rsquared_test=model.score(pred_test,tar_test)
print ('training data R-square')
print(rsquared_train)
print ('test data R-square')
print(rsquared_test)
