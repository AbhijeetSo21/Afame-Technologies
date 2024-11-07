#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Movie Rating Prediction with Python

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

df = pd.read_csv('/Users/abhi/Downloads/Moviedataset.csv', encoding='ISO-8859-1')

print(df.head())


# In[3]:


# Drop the 'Year' and 'Name' columns as they are not significant
df = df.drop(columns=['Name', 'Year'])

print("\nData after dropping 'Name' and 'Year' columns:")
print(df.head())


# In[9]:


# Handle missing values
# Filling missing ratings with the mean rating of the dataset
df['Rating'] = df['Rating'].fillna(df['Rating'].mean())

# Converting 'Votes' to numeric by removing commas and handling any errors
df['Votes'] = df['Votes'].replace({',': ''}, regex=True)  # Remove commas
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')  

df['Duration'] = df['Duration'].str.replace(' min', '', regex=False)  # Remove ' min'
df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')  # Converting to numeric (int or float)

# Filling other missing values with the mode 
df['Genre'] = df['Genre'].fillna(df['Genre'].mode()[0])
df['Votes'] = df['Votes'].fillna(df['Votes'].median())
df['Director'] = df['Director'].fillna(df['Director'].mode()[0])
df['Actor 1'] = df['Actor 1'].fillna(df['Actor 1'].mode()[0])
df['Actor 2'] = df['Actor 2'].fillna(df['Actor 2'].mode()[0])
df['Actor 3'] = df['Actor 3'].fillna(df['Actor 3'].mode()[0])

print("\nData after handling missing values:")
print(df.head())


# In[12]:


df.replace([np.inf, -np.inf], np.nan, inplace=True)  
df.dropna(inplace=True) 
print(df.dtypes)
print(df.head())


# In[13]:


# Encoding for categorical features (like, 'Genre', 'Director', 'Actor')

from sklearn.preprocessing import LabelEncoder

# Instantiate LabelEncoder
label_encoder = LabelEncoder()

# Encoding categorical columns
df['Genre'] = label_encoder.fit_transform(df['Genre'])
df['Director'] = label_encoder.fit_transform(df['Director'])
df['Actor 1'] = label_encoder.fit_transform(df['Actor 1'])
df['Actor 2'] = label_encoder.fit_transform(df['Actor 2'])
df['Actor 3'] = label_encoder.fit_transform(df['Actor 3'])

print("\nData after encoding categorical features:")
print(df.head())


# In[14]:


X = df.drop(columns=['Rating'])  # Features
y = df['Rating']  # Target variable

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Predicting ratings for the test set
y_pred = model.predict(X_test)

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
print("\nMean Squared Error of the model:", mse)


# In[17]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), ['Duration']),  # Impute missing values in Duration with median
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing values in categorical columns with the mode
            ('encoder', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical columns
        ]), ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3'])  # Apply to categorical columns
    ])

# Define the regression model (Random Forest Regressor)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))  # Random Forest Regressor
])

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")


# In[19]:


# Function to predict movie rating based on user input
def predict_rating(genre, director, actor1, actor2, actor3, duration):
    input_data = pd.DataFrame({
        'Genre': [genre],
        'Director': [director],
        'Actor 1': [actor1],
        'Actor 2': [actor2],
        'Actor 3': [actor3],
        'Duration': [duration]
    })

    # Make the prediction using the trained model
    predicted_rating = model.predict(input_data)
    return predicted_rating[0]

# Example usage of the prediction function
print("Example Prediction:")
predicted_rating = predict_rating(
    genre='Drama',
    director='Rajkumar Hirani',
    actor1='Aamir Khan',
    actor2='Kareena Kapoor',
    actor3='Sharman Joshi',
    duration=160
)

print(f"Predicted Movie Rating: {predicted_rating:.2f}")


# In[ ]:




