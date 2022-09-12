#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('student_scores.csv')


# In[3]:


print(df.corr())


# In[4]:


print(df.describe())


# In[5]:


y = df['Scores'].values.reshape(-1, 1)
X = df['Hours'].values.reshape(-1, 1)


# In[6]:


print('X shape:', X.shape)
print('X:', X)


# In[7]:


print(df['Hours'].values) # [2.5 5.1 3.2 8.5 3.5 1.5 9.2 ... ]
print(df['Hours'].values.shape) # (25,)


# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)


# In[9]:


print(y_train)


# In[10]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[11]:


regressor.intercept_


# In[12]:


regressor.coef_


# In[13]:


def calc(slope, intercept, hours):
    return slope*hours+intercept


# In[14]:


score = calc(regressor.coef_, regressor.intercept_, 9.5)
print(score)


# In[15]:


score = regressor.predict([[9.5]])
print(score)


# In[16]:


y_pred = regressor.predict(X_test)


# In[17]:


df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
print(df_preds)


# In[19]:


from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Error medio absoluto: ", mae, "\nError medio cuadrado: ", mse, "\nRaiz del error medio cuadrado: ", rmse)

