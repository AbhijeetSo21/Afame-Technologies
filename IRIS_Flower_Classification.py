#!/usr/bin/env python
# coding: utf-8

# In[8]:


# IRIS Flower Classification

import pandas as pd 
from sklearn.datasets import load_iris       # We can use the dataset for iris flowers by load_iris from sklearn
iris = load_iris()

# iris = pd.read_csv('/Users/abhi/Downloads/irisflowerdata.csv')     # or we can use the csv file data
# iris


# In[9]:


dir(iris)


# In[10]:


iris.feature_names


# In[11]:


iris.target


# In[12]:


iris.target_names


# In[13]:


df = pd.DataFrame(iris.data, columns=iris.feature_names)
df.head()


# In[14]:


df['target'] = iris.target
df


# In[17]:


df[df.target==0].head()


# In[18]:


df[df.target==1].head()


# In[19]:


df[df.target==2].head()


# In[20]:


df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])
df.head()


# In[21]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[22]:


df0 = df[df.target==0]
df1 = df[df.target==1]
df2 = df[df.target==2]


# In[31]:


plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'], color='green', marker='*')      # Setosa
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'], color='blue', marker='.')       # Versicolor
plt.scatter(df2['sepal length (cm)'], df2['sepal width (cm)'], color='red', marker='+')        # Virginica


# In[32]:


plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'], color='orange', marker='+')         # Setosa
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'], color='yellow', marker='*')         # Versicolor
plt.scatter(df2['petal length (cm)'], df2['petal width (cm)'], color='red', marker='+')            # Virginica


# In[33]:


from sklearn.model_selection import train_test_split
x = df.drop(['target', 'flower_name'], axis='columns')
y = df['target']


# In[34]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
len(x_train)


# In[35]:


from sklearn.svm import SVC
model = SVC()
model.fit(x_train, y_train)


# In[36]:


model.score(x_test, y_test)


# In[42]:


model.predict([[5.2, 3.6, 1.3, 0.3]])        # Output : array([0]) (means Setosa)


# In[43]:


model.predict([[7.2, 3.1, 4.8, 1.3]])        # Output : array([1]) (means Versicolor)


# In[45]:


model.predict([[6.1, 3.2, 6.2, 2.4]])        # Output : array([2]) (means Virginica)


# In[ ]:




