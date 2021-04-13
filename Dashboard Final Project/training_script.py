# This script is created based on feature selection result, so some columns are going to be removed and feature selection won't undertaken again 
# Basic Operations
import pandas as pd
import numpy as np

# ML Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Feature Engineering
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import category_encoders as ce
from sklearn.feature_selection import RFE
from imblearn.pipeline import Pipeline

# Evaluation
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler

# Model
import pickle

# data
df=pd.read_csv('C:/Users/user/Desktop/Purwadhika/Final Project/Submit/E Commerce Dataset.csv')
df=df.drop(['CustomerID', 'PreferredLoginDevice', 'WarehouseToHome', 'PreferredPaymentMode', 'HourSpendOnApp', 'OrderAmountHikeFromlastYear'], axis=1)

# preprocess
transformer=ColumnTransformer([('onehot',OneHotEncoder(drop='first'), ['Gender', 'MaritalStatus']),
                   ('binary',ce.BinaryEncoder(), ['PreferedOrderCat']),
                   ('imputer', SimpleImputer(strategy='mean'), ['Tenure', 'CouponUsed', 'DaySinceLastOrder', 'OrderCount']),
                   ('scaler', StandardScaler(), ['CashbackAmount'])
                              ])

# Data Splitting
X=df.drop(columns='Churn')
y=df['Churn']

# Model Selection
rf=RandomForestClassifier()
under=RandomUnderSampler()

estimator=Pipeline([('transformer', transformer),
                    ('balance', under),
                    ('model', rf)])

# Evaluation
hyperparam_space = {
        'model__criterion' : ['entropy','gini'],
        'model__min_samples_leaf' : [1,5,10,15,20],
        'model__min_samples_split' : [2,5,10,15,20],
        'model__max_depth' : [5,6,7,9,11,13,15]
        }

skfold=StratifiedKFold(n_splits=5)

grid=GridSearchCV(estimator,
                   param_grid=hyperparam_space,
                   cv=skfold,
                   scoring='recall',
                   n_jobs=-1)

grid.fit(X,y)

# Saving Model
grid.best_estimator_.fit(X,y)

file_name='ModelChurnFinal.sav'

pickle.dump(grid.best_estimator_, open(file_name, 'wb'))