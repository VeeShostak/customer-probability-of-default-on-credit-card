
# In[1]:

# ===========================
# Data Preprocessing
# ===========================

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


'''
Dataset Information:
    Data Set Characteristics: Multivariate
    Attribute Characteristics: Integer, Real
    Number of Attributes: 24
    Number of Instances: 30000
    Source: 
        UCI Machine Learning Repository
        institutions: (1) Department of Information Management, Chung Hua University, Taiwan. (2) Department of Civil Engineering, Tamkang University, Taiwan. 
    

Dataset Attributes:
    This research employed a binary variable, default payment (Yes = 1, No = 0), as the response variable. 
    This study reviewed the literature and used the following 23 variables as explanatory variables: 
        
    X1: Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit. 
    X2: Gender (1 = male; 2 = female). 
    X3: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others). 
    X4: Marital status (1 = married; 2 = single; 3 = others). 
    X5: Age (year). 
    X6 - X11: History of past payment. We tracked the past monthly payment records (from April to September, 2005) as follows: X6 = the repayment status in September, 2005; X7 = the repayment status in August, 2005; . . .;X11 = the repayment status in April, 2005. The measurement scale for the repayment status is: -1 = pay duly(paid appropriately); 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above. 
    X12-X17: Amount of bill statement (NT dollar). X12 = amount of bill statement in September, 2005; X13 = amount of bill statement in August, 2005; . . .; X17 = amount of bill statement in April, 2005. 
    X18-X23: Amount of previous payment (NT dollar). X18 = amount paid in September, 2005; X19 = amount paid in August, 2005; . . .;X23 = amount paid in April, 2005.
'''


# Import the dataset
# Note we have two headers in dataset, skip the extra row
dataset = pd.read_excel('dataset/default-of-credit-card-clients.xls', skiprows=1)


# In[2]:

# matrix of features (independent variables). 
# select: all rows. columnn indices 1 - 23 (exclude columns: id#)
X = dataset.iloc[:, 1:24].values 

# matrix of dependent vars (output to predict)
# select: all rows. last column
y = dataset.iloc[:, -1].values



# In[3]:
# Encode categorical data (converting into numbers to make it comparable for 
# Regression) and create dummy vars if needed(avoid dummy var trap)

# This step is not needed since our dataset has no categorical variables


# In[4]:

# Split the dataset into the Training set and Test set
# Train:  0.8 (80%) of obesrvations to train ANN.
# Test: 0.2 (20%) of obesrvations to test ANN. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[5]:
# Feature Scaling to standardize the range of independent variables or features of data (normalize).
# we want to have standard normally distributed data and dont want to have one independent var dominating another one.
# Also due to intensive calculations and parallel execution, scaling eases these calculations.

# If one of the features has a broad range of values, the distance will be governed by
# this particular feature. Therefore, the range of all features should be normalized so 
# that each feature contributes approximately proportionately to the final distance.

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
'''
    fit_transform vs transform: https://datascience.stackexchange.com/questions/12321/difference-between-fit-and-fit-transform-in-scikit-learn-models
(you have to use the same two parameters μ and σ (values) that you used for centering the training set)
sklearn's transform's fit() just calculates the parameters (e.g. μ and σ)
and saves them as an internal objects state. 
Afterwards, you can call its transform() method to apply the transformation to a particular set of examples.
'''
X_train = sc.fit_transform(X_train) # Fit to data, then transform it.
X_test = sc.transform(X_test) # Perform standardization by centering and scaling

