#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv('D:\logic_insurence.csv')
df.head()


# In[4]:


plt.scatter(df.age,df.bought_insurence,marker='+',color='red') # scatter plot to see data


# In[5]:


df.shape


# In[9]:


from sklearn.model_selection import train_test_split # importing library for train test


# In[17]:


X_train,X_test,y_train,y_test = train_test_split(df[['age']],df.bought_insurence,test_size=0.1) # test is test data 0.1


# In[18]:


X_test # data set we are using to test


# In[19]:


X_train # data set using for train


# In[20]:


from sklearn.linear_model import LogisticRegression # importing logistic regression


# In[21]:


model = LogisticRegression()


# In[22]:


model.fit(X_train,y_train)  # fiting to train


# In[23]:


model.predict(X_test) #  predicting in testing data first person will not buy a insuerence but rest will buy


# In[24]:


model.score(X_test,y_test) # model is 66% acurate


# In[42]:


model.predict_proba(X_test) # first column shows no and sec shows yes , in sec column 9.97 & 9.99 showing yes for insuerence.


# In[43]:


import math     # Lets defined sigmoid function 
def sigmoid(x):
  return 1 / (1 + math.exp(-x))


# In[44]:


def prediction_function(age):
    z = 0.042 * age - 1.53 # 0.04150133 ~ 0.042 and -1.52726963 ~ -1.53
    y = sigmoid(z)
    return y


# In[45]:


age = 43                  # 0.568 is more than 0.5 which means person with 43 age will  buy insurance
prediction_function(age)


# In[41]:


age = 29             # less likely to buy insurence because 0.4 is less then 0.5
prediction_function(age)


# In[ ]:




