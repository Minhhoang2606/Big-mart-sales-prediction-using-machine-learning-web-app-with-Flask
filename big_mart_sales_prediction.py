# Data source: https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data/data

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score

# Data preprocessing
big_mart_df = pd.read_csv('Train.csv')
big_mart_df.head()
big_mart_df.info()
big_mart_df.isnull().sum()
'''
- There are a lot of missing values in the 'Item_Weight' and 'Outlet_Size' columns.
- We will fill in the missing values of 'Item_Weight' with the mean of the column since it's a numerical column, and fill in the missing values of 'Outlet_Size' with the mode of the column since it's a categorical column.
'''
## Handle missing values of 'Item_Weight' by filling in with the mean
big_mart_df['Item_Weight'] = big_mart_df['Item_Weight'].fillna(big_mart_df['Item_Weight'].mean())

big_mart_df.isnull().sum()
## Handle missing values of 'Outlet Size'
big_mart_df['Outlet_Size'].unique()
big_mart_df['Outlet_Size'].mode()
'''
Since Outlet_Size depends on Outlet_Type, we will replace the missing values of Outlet_Size with the mode of it in each Outlet_Type. So the data will be more accurate.
'''
big_mart_df['Outlet_Type'].unique()

mode_of_outlet_size = big_mart_df.pivot_table(values = 'Outlet_Size', columns = 'Outlet_Type', aggfunc = (lambda x: x.mode()[0]))
## Fill in missing values of 'Outlet_Size' 
missing_value = big_mart_df['Outlet_Size'].isnull()
big_mart_df.loc[missing_value, 'Outlet_Size'] = big_mart_df.loc[missing_value, 'Outlet_Type'].apply(lambda x: mode_of_outlet_size[x])

big_mart_df.isnull().sum()

big_mart_df.describe()

# EDA
## Histogram of numerical features
num_cols = big_mart_df.select_dtypes(include = [np.number]).columns
fig, ax = plt.subplots(nrows = len(num_cols), ncols = 1, figsize = (8, 6 * len(num_cols)))
for i, col in enumerate(num_cols):
    sns.histplot(big_mart_df[col], ax = ax[i], bins = 30, edgecolor = 'black', kde = True)
    ax[i].set_title(f'Histogram of {col}')
plt.tight_layout()
plt.show()

## Countplot of categorical features
### Outlet_Establishment_Year
## Countplot of Outlet_Establishment_Year
plt.figure(figsize = (8, 6))
sns.countplot(x = 'Outlet_Establishment_Year', data = big_mart_df, palette = 'viridis')
for p in plt.gca().patches:
    plt.text(x = p.get_x() + p.get_width() / 2, y = p.get_height(), s = f'{int(p.get_height())}', ha = 'center', va = 'bottom')
plt.title('Countplot of Outlet_Establishment_Year')
plt.xlabel('Outlet_Establishment_Year')
plt.ylabel('Count')
plt.show()

## Countplot of Item_Fat_Content
plt.figure(figsize = (8, 6))
sns.countplot(x = 'Item_Fat_Content', data = big_mart_df, palette = 'viridis')
for p in plt.gca().patches:
    plt.text(x = p.get_x() + p.get_width() / 2, y = p.get_height(), s = f'{int(p.get_height())}', ha = 'center', va = 'bottom')
plt.title('Countplot of Item_Fat_Content')
plt.xlabel('Item_Fat_Content')
plt.ylabel('Count')
plt.show()
'''
We have 'Low Fat' in capital letters and 'low fat' in small letters, so as "Regular' and 'reg', Which are dupplicates. We will fix that later.
'''
## Countplot of Item_Type
plt.figure(figsize = (25, 6))
sns.countplot(x = 'Item_Type', data = big_mart_df, palette = 'viridis')
for p in plt.gca().patches:
    plt.text(x = p.get_x() + p.get_width() / 2, y = p.get_height(), s = f'{int(p.get_height())}', ha = 'center', va = 'bottom')
plt.title('Countplot of Item_Type')
plt.xlabel('Item_Type')
plt.ylabel('Count')
plt.show()

## Countplot of Outlet_Identifier
plt.figure(figsize = (8, 6))
sns.countplot(x = 'Outlet_Identifier', data = big_mart_df, palette = 'viridis')
for p in plt.gca().patches:
    plt.text(x = p.get_x() + p.get_width() / 2, y = p.get_height(), s = f'{int(p.get_height())}', ha = 'center', va = 'bottom')

## Countplot of Outlet_Size
plt.figure(figsize = (8, 6))
sns.countplot(x = 'Outlet_Size', data = big_mart_df, palette = 'viridis')
for p in plt.gca().patches:
    plt.text(x = p.get_x() + p.get_width() / 2, y = p.get_height(), s = f'{int(p.get_height())}', ha = 'center', va = 'bottom')

'''
There is no issue with other categorical features. We will fix the 'low fat' duplicated issue only.
'''
big_mart_df.replace({'Item_Fat_Content': {'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'}}, inplace = True)

# Model training
## Label encoding
le = LabelEncoder()

for col in big_mart_df.select_dtypes(include=['object']).columns:
    big_mart_df[col] = le.fit_transform(big_mart_df[col])

big_mart_df.head()

## Splitting the data into train and test
X = big_mart_df.drop(columns = ['Item_Outlet_Sales'])
y = big_mart_df['Item_Outlet_Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

model = XGBRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

## Model evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f'RMSE: {rmse:.2f}')
print(f'R2 score: {r2:.2f}')

'''
The RMSE (Root Mean Squared Error) of 1161.00 indicates the average magnitude of the errors made by your model in predicting the target variable. In this case, it means that the model's predictions are 1161.00 units away from the actual values on average. 

The R2 score of 0.52 indicates the proportion of the variance in the target variable that is explained by the model. In this case, it means that the model accounts for only about 52% of the variation in the target variable. This suggests that the model is not very good at predicting the target variable, as it does not capture a significant portion of the variability in the data.
'''

## Model interpretation
importance = model.feature_importances_
indices = np.argsort(importance)[::-1]
plt.figure(figsize = (8, 6))
plt.bar(range(X.shape[1]), importance[indices], align = 'center')
plt.xticks(range(X.shape[1]), X.columns[indices], rotation = 90)
plt.xlabel('Features')
plt.ylabel('Feature Importance')
plt.show()

# Model deployment
import pickle
pickle.dump(model, open('model.pkl', 'wb'))