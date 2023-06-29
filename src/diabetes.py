#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.datasets import load_diabetes


# In[3]:


# Load the diabetes dataset
diabetes = load_diabetes()


# In[4]:


diabetes.keys()


# In[5]:


print(diabetes.DESCR)


# In[6]:


diabetes.feature_names


# In[7]:


dataset = pd.DataFrame(diabetes.data,columns=diabetes.feature_names)


# In[8]:


dataset


# In[9]:


diabetes.target


# In[10]:


dataset['target']=diabetes.target


# In[11]:


dataset


# In[12]:


dataset.info()


# In[13]:


dataset.describe()


# In[14]:


dataset.corr()


# In[15]:


dataset.isnull().sum()


# In[16]:


import seaborn as sns


# In[17]:


sns.pairplot(dataset)


# In[18]:


dataset.corr()


# In[19]:


sns.pairplot(dataset)


# In[20]:


sns.regplot(x="age",y="target",data=dataset)


# In[21]:


plt.plot(dataset['age'],dataset['target'])


# In[22]:


X=dataset.iloc[:,:-1]
Y=dataset.iloc[:,-1]


# In[23]:


X.head()


# In[24]:


Y


# In[25]:


#important step number 1
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)


# In[26]:


X_train


# In[27]:


Y_train


# In[28]:


#important step number 2
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[29]:


X_train=scaler.fit_transform(X_train)


# In[30]:


X_test=scaler.transform(X_test)


# In[31]:


import pickle
pickle.dump(scaler,open('scaling1.pkl','wb'))


# # model Traning

# In[32]:


from sklearn.linear_model import LinearRegression


# In[33]:


regression=LinearRegression()


# In[34]:


regression.fit(X_train,Y_train)


# In[35]:


print(regression.coef_)


# In[36]:


print(regression.intercept_)


# In[37]:


## on which parameters the model has been trained
regression.get_params()


# In[38]:


## Prediction With Test Data
reg_pred=regression.predict(X_test)


# In[39]:


reg_pred


# In[40]:


plt.scatter(Y_test,reg_pred)


# In[ ]:





# In[43]:


residuals=Y_test-reg_pred


# In[44]:


residuals


# In[45]:


sns.displot(residuals,kind="kde")


# In[46]:


plt.scatter(reg_pred,residuals)


# In[47]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(Y_test,reg_pred))
print(mean_squared_error(Y_test,reg_pred))
print(np.sqrt(mean_squared_error(Y_test,reg_pred)))


# In[48]:


from sklearn.metrics import r2_score
score=r2_score(Y_test,reg_pred)
print(score)


# In[49]:


#display adjusted R-squared
1 - (1-score)*(len(Y_test)-1)/(len(Y_test)-X_test.shape[1]-1)


# In[ ]:




