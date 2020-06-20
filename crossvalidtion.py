#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


data_prep=pd.read_csv("decisiontreeAdultincome.csv")
data_prep.head()


# In[4]:


data_prep=pd.get_dummies(data_prep,drop_first=True)


# In[5]:


X=data_prep.iloc[:,:-1]
Y=data_prep.iloc[:,-1]


# In[11]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=1234)


# In[7]:


# Import and train Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=1234)


# In[8]:


# Import and train Support Vector Classifier
from sklearn.svm import SVC
svc = SVC(kernel='rbf', gamma=0.5)


# In[22]:


from sklearn.linear_model import LogisticRegression
lrc = LogisticRegression(random_state=1234)


# In[23]:


# perform cross validation


# In[24]:


from sklearn.model_selection import cross_validate


# In[25]:


cv_result_dtc=cross_validate(dtc,X,Y,cv=10,return_train_score=True)
cv_result_rfc=cross_validate(rfc,X,Y,cv=10,return_train_score=True)
cv_result_svc=cross_validate(svc,X,Y,cv=10,return_train_score=True)
cv_result_lrc=cross_validate(lrc,X,Y,cv=10,return_train_score=True)


# In[26]:


cv_result_dtc


# In[18]:


cv_result_rfc


# In[19]:


cv_result_svc


# In[20]:


cv_result_lrc


# In[27]:


#get average of results
import numpy as np


# In[38]:


dtc_test_average = np.average(cv_result_dtc['test_score'])
rfc_test_average = np.average(cv_result_rfc['test_score'])
svc_test_average = np.average(cv_result_svc['test_score'])
lrc_test_average = np.average(cv_result_lrc['test_score'])


# In[39]:


dtc_test_average


# In[40]:


dtc_train_average = np.average(cv_result_dtc['train_score'])
rfc_train_average = np.average(cv_result_rfc['train_score'])
svc_train_average = np.average(cv_result_svc['train_score'])
lrc_train_average = np.average(cv_result_lrc['train_score'])


# In[43]:


# print the results 
print()
print()
print('        ','Decision Tree  ', 'Random Forest  ','Support Vector   ','logistic Regression')
print('        ','---------------', '---------------','-----------------','-------------------')
print('Test  : ',
      round(dtc_test_average, 4), '        ',
      round(rfc_test_average, 4), '        ',
      round(svc_test_average, 4), '        ',
      round(lrc_test_average, 4))
print('Train : ',
      round(dtc_train_average, 4), '        ',
      round(rfc_train_average, 4), '        ',
      round(svc_train_average, 4), '        ',
      round(lrc_test_average, 4))




# - the difference between the support vector test,train values are less compared to others.

# In[ ]:





# In[ ]:




