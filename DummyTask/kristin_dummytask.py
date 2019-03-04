#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 16:57:10 2019

@author: kristinblesch
"""
#import os
#os.chdir('/Users/kristinblesch/Desktop/ETH ')

import pandas as pd
df_train = pd.read_csv('train.csv')
 
# using Regression
from sklearn.linear_model import LinearRegression
x_train = df_train[df_train.columns[2:12]]
y_train = df_train['y']
model = LinearRegression().fit(x_train, y_train)

df_test = pd.read_csv('test.csv')
x_test = df_test[df_test.columns[1:11]]
y_pred = model.predict(x_test)

result = pd.DataFrame({'Id': df_test['Id'],'y_pred': y_pred})

result.to_csv('submission.csv')
