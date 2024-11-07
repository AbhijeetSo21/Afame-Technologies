#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Sales Prediction Using Python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('/Users/abhi/Downloads/Sales.csv') 

print(data.head())


# In[2]:


print(data.isnull().sum())


# In[3]:


correlation_matrix = data.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


# In[4]:


X = data[['TV', 'Radio', 'Newspaper']] 
y = data['Sales']  

# Split the data into 80% training, 20% testing

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[5]:


# Standardizing the data

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()


# In[6]:


model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R-squared (R2): {r2}')


# In[7]:


# Visualize the actual vs predicted sales
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.show()


# In[8]:


print("\nCoefficients of the Linear Model:")
print(f'TV: {model.coef_[0]}')
print(f'Radio: {model.coef_[1]}')
print(f'Newspaper: {model.coef_[2]}')
print(f'Intercept: {model.intercept_}')


# In[9]:


# Predicting future sales 
new_ad_expenditure = pd.DataFrame({
    'TV': [250.0, 100.0],  
    'Radio': [30.0, 40.0],  
    'Newspaper': [50.0, 10.0]  
})

new_ad_expenditure_scaled = scaler.transform(new_ad_expenditure)  # Scaling the new data
predicted_sales = model.predict(new_ad_expenditure_scaled)

print("\nPredicted Sales for new advertising expenditure:")
print(predicted_sales)


# In[ ]:




