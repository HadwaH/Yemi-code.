#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import classification_report

# Load the data
trades = pd.read_excel(r'invesco_complete.xlsx', engine='openpyxl')


# In[17]:


trades.columns


# In[18]:


trades


# In[19]:


trades.groupby('Dealer').agg({'ISIN':'count','Screen saving (price)': 'mean','Accepted Vol':'mean'})


# In[20]:


trades.groupby('Dealer').agg({'ISIN':'count','Screen saving (price)': 'mean'}).sort_values(by = 'Screen saving (price)')


# In[21]:


trades[trades.Dealer =='DB']


# In[22]:


# Drop rows with missing values
trades.dropna(subset=['Dealer', 'Screen Saving (amount)'], inplace=True)


# In[23]:


# Define feature columns
feature_columns = ['Time To Quote', 'Accepted Vol']


# In[24]:


# Create a dictionary to store cheapest screen saving amounts for each dealer
dealer_cheapest_amounts = {}


# In[25]:


# Iterate over unique dealers
for dealer in trades['Dealer'].unique():
   # Filter data for the specific dealer
   dealer_data = trades[trades['Dealer'] == dealer]

   # Check if the dealer has sufficient data points
   if len(dealer_data) > 1:  # At least 2 data points are required for splitting
       # Separate features and target variable
       X = dealer_data[feature_columns]
       y = dealer_data['Screen Saving (amount)']

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
       xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

       # Fit the XGBoost model
       xgb_regressor.fit(X_train_numerical, y_train)

       # Make predictions on the test data
       y_pred = xgb_regressor.predict(X_test_numerical)

       # Calculate the cheapest screen saving amount for the dealer
       cheapest_amount = np.min(y_pred)
       dealer_cheapest_amounts[dealer] = cheapest_amount


# In[26]:


# Rank dealers based on cheapest amounts
ranked_dealers = sorted(dealer_cheapest_amounts, key=lambda x: dealer_cheapest_amounts[x])


# In[27]:


# Print ranked dealers
for rank, dealer in enumerate(ranked_dealers, start=1):
   print(f"Rank {rank}: Dealer '{dealer}' with cheapest amount of {dealer_cheapest_amounts[dealer]}")


# In[28]:


# Create a transformer for numerical features
numerical_transformer = StandardScaler()


# In[29]:


# Transform all data for permutation importance
X_numerical = numerical_transformer.fit_transform(trades[feature_columns])
y_amount = trades['Screen Saving (amount)']


# In[30]:


# Create an XGBoost regressor for amount prediction
xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)


# In[31]:


# Fit the XGBoost model
xgb_regressor.fit(X_numerical, y_amount)


# In[32]:


from sklearn.inspection import permutation_importance  # Add this import


# In[33]:


import eli5
from eli5.sklearn import PermutationImportance


# In[34]:


import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import plot_partial_dependence
from sklearn.inspection import permutation_importance  # Add this import
import matplotlib.pyplot as plt

# Load the data
trades = pd.read_excel('invesco_complete.xisx', engine='openpyxl')

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

# Calculate permutation importance
perm_importance = permutation_importance(xgb_regressor, X_train_numerical, y_train, n_repeats=30, random_state=42)


# In[35]:


pip install -U scikit-learn 


# In[36]:


# Calculate permutation importance
perm_importance = permutation_importance(xgb_regressor, X_numerical, y_amount, n_repeats=30, random_state=42)


# In[37]:


# Print permutation importance scores
for i, feature in enumerate(feature_columns):
   print(f'{feature} importance: {perm_importance.importances_mean[i]}')


# In[38]:


# Plot permutation importance scores using Seaborn
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.barplot(x=perm_importance.importances_mean, y=feature_columns)
plt.xlabel('Permutation Importance')
plt.ylabel('Features')
plt.title('Permutation Importance for Predicting Cheapest Amount')
plt.show()


# In[5]:


import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import eli5
from eli5.sklearn import PermutationImportance

# Load the data
trades = pd.read_excel('invesco_complete.xlsx', engine='openpyxl')

# Drop rows with missing values
trades.dropna(subset=['Dealer', 'Screen Saving (amount)'], inplace=True)

# Define feature columns
feature_columns = ['Time To Quote', 'Accepted Vol']

# Create a dictionary to store cheapest screen saving amounts for each dealer
dealer_cheapest_amounts = {}

# Iterate over unique dealers
for dealer in trades['Dealer'].unique():
   # Filter data for the specific dealer
   dealer_data = trades[trades['Dealer'] == dealer]

   # Check if the dealer has sufficient data points
   if len(dealer_data) > 1:  # At least 2 data points are required for splitting
       # Separate features and target variable
       X = dealer_data[feature_columns]
       y = dealer_data['Screen Saving (amount)']

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
           n_estimators=1000,  # Set a large number of estimators
           learning_rate=0.1,  # Initial learning rate
           max_depth=5,        # Example hyperparameters, tune these
           min_child_weight=1,
           gamma=0,
           subsample=0.8,
           colsample_bytree=0.8,
           verbosity=0
       )

       # Fit the XGBoost model
       xgb_regressor.fit(X_train_numerical, y_train)

       # Make predictions on the test data
       y_pred = xgb_regressor.predict(X_test_numerical)

       # Calculate the cheapest screen saving amount for the dealer
       cheapest_amount = np.min(y_pred)
       dealer_cheapest_amounts[dealer] = cheapest_amount

# Rank dealers based on cheapest amounts
ranked_dealers = sorted(dealer_cheapest_amounts, key=lambda x: dealer_cheapest_amounts[x])

# Print ranked dealers
for rank, dealer in enumerate(ranked_dealers, start=1):
   print(f"Rank {rank}: Dealer '{dealer}' with cheapest amount of {dealer_cheapest_amounts[dealer]}")

# Fit XGBoost model on all data for permutation importance
X_numerical = numerical_pipeline.transform(trades[feature_columns])
y_amount = trades['Screen Saving (amount)']

xgb_regressor.fit(X_numerical, y_amount)

# Calculate permutation importance using eli5
perm_importance = PermutationImportance(xgb_regressor, random_state=42).fit(X_numerical, y_amount)

# Display permutation importance using eli5
eli5.show_weights(perm_importance, feature_names=feature_columns)


# We measure the amount of randomness in our permutation importance calculation by repeating the process with multiple shuffles. The number after the ± measures how performance varied from one-reshuffling to the next.

# While feature importance shows what variables most affect predictions, partial dependence plots show how a feature affects predictions.
# 
# 

# In[39]:


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


# In[40]:


from sklearn.inspection import plot_partial_dependence


# In[41]:


pip install plot_partial_dependence


# In[42]:


pip install -U scikit-learn --user


# In[43]:


pip install --upgrade scikit-learn


# In[23]:


trades.iloc[0]


# In[7]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
trades = pd.read_excel('invesco_complete.xlsx', engine='openpyxl')

# Drop rows with missing 'Dealer' values
trades = trades.dropna(subset=['Dealer'])

# Create a countplot using Seaborn
plt.figure(figsize=(10, 6))
sns.countplot(x='Dealer', data=trades, order=trades['Dealer'].value_counts().index)
plt.xticks(rotation=45, ha='right')
plt.xlabel('Dealer')
plt.ylabel('Number of Trades')
plt.title('Number of Trades per Dealer')
plt.tight_layout()

# Show the plot
plt.show()


# In[35]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
trades = pd.read_excel('invesco_complete.xlsx', engine='openpyxl')

# Drop rows with missing 'Dealer' values
trades = trades.dropna(subset=['Dealer'])

# Create a scatter plot using Seaborn
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Cover Price', y='Dealer', data=trades)
plt.xlabel('Screen Saving Amount')
plt.ylabel('Dealer')
plt.title('Screen Saving Amount vs. Dealer')
plt.tight_layout()

# Show the plot
plt.show()


# In[10]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
trades = pd.read_excel('invesco_complete.xlsx', engine='openpyxl')

# Drop rows with missing 'Sector' and 'Accepted Vol' values
trades = trades.dropna(subset=['Sector', 'Accepted Vol'])

# Group the data by sector and sum the accepted volume
sector_accepted_vol = trades.groupby('Sector')['Accepted Vol'].sum()

# Create a pie chart using Matplotlib
plt.figure(figsize=(10, 6))
plt.pie(sector_accepted_vol, labels=sector_accepted_vol.index, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Distribution of Accepted Volume by Sector')
plt.tight_layout()

# Show the plot
plt.show()


# In[17]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
trades = pd.read_excel('invesco_complete.xlsx', engine='openpyxl')

# Drop rows with missing 'Sector' and 'Accepted Vol' values
trades = trades.dropna(subset=['Sector', 'Accepted Vol'])

# Group the data by sector and sum the accepted volume
sector_accepted_vol = trades.groupby('Sector')['Accepted Vol'].sum()

# Create a pie chart using Matplotlib
plt.figure(figsize=(10, 6))
colors = sns.color_palette('pastel')
plt.pie(sector_accepted_vol, labels=sector_accepted_vol.index, autopct='%1.1f%%', startangle=140, colors=colors)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Distribution of Accepted Volume by Sector')

# Add a legend with colors and descriptions
legend_labels = [f'{sector} ({vol:.1f}M)' for sector, vol in zip(sector_accepted_vol.index, sector_accepted_vol / 1e6)]
plt.legend(legend_labels, title='Sectors', loc='upper right')

plt.tight_layout()

# Show the plot
plt.show()

