
# coding: utf-8

# # Introduction to Machine Learning: Dummy Task
# 
# __Author__: Jannick Sicher

# ### Initial Configurations

# In[1]:


import pandas as pd
import sklearn as sk
import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


# ### Train and Test Set

# In[2]:


### Load train and test set
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')


# In[3]:


## Display train data
df_train.head()


# In[4]:


## Display test data
df_test.head()


# In[5]:


## Train data: Predictors
df_train_X = df_train[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']]
df_train_X.head()


# In[6]:


## Train data: Target
df_train_Y = df_train[['y']]
df_train_Y.head()


# In[7]:


## Test data: Predictors
df_test_X = df_test[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']]
df_test_X.head()


# ### Calculating Mean of Target Variable in Test Set

# In[8]:


## Test data: Target Variable
df_test_Y1 = df_test[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']]
df_test_Y1['sum']= df_test_Y1.iloc[:,].sum(axis=1)
df_test_Y1['y'] = df_test_Y1["sum"]/10
df_test_Y1.head()


# ### Linear Regression

# In[9]:


lm = LinearRegression()
lm.fit(df_train_X, df_train_Y)


# In[10]:


df_test['y'] = lm.predict(df_test_X)


# In[11]:


sns.regplot(df_test_Y1['y'],df_test['y'], fit_reg=False)


# ### Model Evaluation

# In[12]:


# model evaluation
RMSE = mean_squared_error(df_test_Y1['y'], df_test['y'])**0.5
r2 = r2_score(df_test_Y1['y'], df_test['y'])

# printing values
print('Slope:' ,lm.coef_)
print('Intercept:', lm.intercept_)
print('Root mean squared error: ', RMSE)
print('R2 score: ', r2)


# ### Submission File

# In[13]:


submission = df_test[['Id', 'y']]
submission.head()


# In[14]:


submission.to_csv('submission.csv')    #to save the dataframe, df to submission.csv

