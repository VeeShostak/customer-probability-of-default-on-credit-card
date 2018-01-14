'''
            Artificial Neural Network for Classification
Used to solve a data analytics challenge for a bank looking to determine consumer credit risk.

This is an Artificial Neural Network that can predict, based on 24 attributes of
a customer, if an individual customer will default on their payment next month for their credit card.

In addition, we are be able to rank all the customers of the bank, based on 
their probability of default. To do that, we use the right Deep Learning model, 
one that is based on a probabilistic approach (need the output layer(dependent variable)
to use the sigmoid activation function
'''

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


# In[6]:
# Intitialize the ANN

# ===========================
# Make the ANN
# ===========================

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential # to initialize ANN
from keras.layers import Dense # to create layers in ANN

# Initialising the ANN (we will define it as a sequence of layers or you can define a graph)
classifier = Sequential()
'''
1. Dense() function will randomly initialize the weights to small numbers close to 0 (but not 0)
2. place each feature in one input node e.g: 11 input vars => 11 input nodes
3. farward propagation pass inputs from left to right where the neuron will be activated in way
   where impact of each neurons activation is limited by the weights. (sum allweights, apply activation function)
   using rectifier func for input layer and sigmoid for outer layer(to get probabilities of customer leaving or staying) 
4. compare predicted result to actual
5. back propagation from right to left. update the weights according to how much they are responsible for he error
   The learnign rate decides by how much we update the weights
6. repeat steps 1-5 and update weights after each observation (row) (reinforcement learning)
   Or: repeat steps 1-5 and updatee the weights only after a batch of observations (Batch learning)
7. When the whole training (all rows of data) passed through the ANN that makes an epoch. Redo more epochs
'''
# use stochastic gradient decscent


#Choose Number of nodes in hidden layer: as average number of nodes in hidden and output layer
#Or experiment with parameter tuning, ross validation techniques
# (23 + 1)/2 for hidden layer

# initialize weights randomly close to 0: kernel_initializer = 'uniform'

# using rectifier activation func for input layter and sigmoid for outer layer(to get probabilities) 


# Add the input layer AND the first hidden layer. (for initial layer specify the input nodes since we have no prev layer)
classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu', input_dim = 23))

# Add the second hidden layer (use prev layer as input nodes)
classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu'))

# Add the output layer (NOTE: if 3 encoded categories for dependent variable need 3 nodes and softmax activator func)
# choose sigmoid just like in logistic regression
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))


# In[7]:
# Compile the ANN (apply Stochastic gradient descent for back propagation)
# Stochastic gradient descent algorithm = adam. 
# loss function within adam alg, based on loss function that we need to optimize to find optimal weights 

# (for simple linear regression loss func is sum of squared errors. 
# in ML(perceptron NN using sigmoid activation function you obtain a logistic regression model):
# looking into stochastic gradient descent loss func is NOT sum of squared errors but is a 
# logarithmic function called logarithmic loss )

# so use binary logarithmic loss func b/c (binary_entropy = dependent var has binary outcome, if i categories = categorical_crossentropy)
# criteria to evaluate our model metrics = ['accuracy'] (after weights updated, algo uses accuracy criterion to improve models performance) (when we fit accuracy will increase little by litle until reach top accuracy since we chose accuracy metric)
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[8]:
# Fit the ANN to the Training set (experimment with batch size and epochs)
# batch_size = 10, epochs = 100
# loss: 0.4234 - acc: 0.8193
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)


# In[9]:
# save the model

'''
 Model weights are saved to HDF5 format.(grid format that is ideal for storing multi-dimensional arrays of numbers.)
 HDF5 lets you store huge amounts of numerical data, and easily manipulate that data from NumPy.
 
 The model structure can be described and saved using two different formats: JSON and YAML.
 '''

from keras.models import model_from_json

# serialize model to JSON
model_json = classifier.to_json()
with open("saved-model/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("saved-model/model.h5")
print("Saved model to disk")

# In[10]:
# Load the model

# load json and create model
json_file = open('saved-model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("saved-model/model.h5")
print("Loaded model from disk")

# compile loaded model
loaded_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# In[11]:
# make predictions

# Predict the Test set results
y_pred = loaded_model.predict(X_test) # gives probability of defaulting next month

y_pred = (y_pred > 0.5) # choose threshold to convert to true or false

# Make the Confusion Matrix to evaluate
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

correct = cm[0][0] + cm[1][1]
incorrect = cm[0][1] + cm[1][0]
accuracy = (correct) / 6000 # tested on number of observations (0.2 * observationsInDataset)
# 82%
