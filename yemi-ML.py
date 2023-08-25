import pandas as pd

import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.inspection import plot_partial_dependence

import matplotlib.pyplot as plt

 

# Load the data

trades = pd.read_excel('invesco_complete.xlsx', engine='openpyxl')

 

# Drop rows with missing values

trades.dropna(subset=['Dealer', 'Screen saving (amount)', 'Time To Quote', 'Accepted Vol'], inplace=True)

 

# Define feature columns

feature_columns = ['Time To Quote', 'Accepted Vol']

 

# Separate features and target variable

X = trades[feature_columns]

y = trades['Screen saving (amount)']

 

# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

 

# Create a transformer for numerical features

numerical_transformer = StandardScaler()

 

# Create a pipeline for numerical features

numerical_pipeline = Pipeline(steps=[

   ('num', numerical_transformer)

])

 

# Fit and transform the numerical features

X_train_numerical = numerical_pipeline.fit_transform(X_train)

X_test_numerical = numerical_pipeline.transform(X_test)

 

# Create an XGBoost regressor for amount prediction

xgb_regressor = xgb.XGBRegressor(

   objective='reg:squarederror',

   random_state=42,

   n_estimators=1000,

   learning_rate=0.1,

   max_depth=5,

   min_child_weight=1,

   gamma=0,

   subsample=0.8,

   colsample_bytree=0.8,

   verbosity=0

)

 

# Fit the XGBoost model

xgb_regressor.fit(X_train_numerical, y_train)

 

# Create and display partial dependence plots

plot_partial_dependence(xgb_regressor, X_train_numerical, features=[1], grid_resolution=50)  # 1 corresponds to "Accepted Vol"

plt.show()
