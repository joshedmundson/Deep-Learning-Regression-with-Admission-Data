import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

import tensorflow as tf
from tensorflow	import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.metrics import r2_score
from sklearn.compose import ColumnTransformer

'''
The aim of this neural network is to predict a student's probability of acceptance to a university based on the following features:
Serial No.,GRE Score, TOEFL Score, University Rating, SOP, LOR ,CGPA, Research
'''

'''
Step 1: Importing and formatting the data
'''
# Load the data 
dataset = pd.read_csv("admissions_data.csv")

# Split the data into features and labels (features used to predict the labels) 
features = dataset.iloc[:, :-1]
labels = dataset.iloc[:, -1]

# Make sure all data is converted to a numerical value (one-hot encoding)
features = pd.get_dummies(features) 

# Split the data into training and testing data 
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Normalise data to improve training efficiency by instansiating and using a ColumnTransformer object
numerical_features = features.select_dtypes(include=['int64', 'float64']) # Returns a dataframe consisting of values meeting the select criteria
numerical_columns = numerical_features.columns
ct = ColumnTransformer([("Normalise", Normalizer(), numerical_columns)], remainder='passthrough')
features_train_scaled = ct.fit_transform(features_train)
features_test_scaled = ct.transform(features_test)


'''
Step 2: Creating the neural network model
'''

# Create sequential neural network with keras
model = Sequential()

# Create and add an input layer with a node for every feature
input_layer = layers.InputLayer(len(features.columns)-1)
model.add(input_layer)

# Add One hidden layer with relu activation 
model.add(layers.Dense(64, activation='relu'))

# Add a one-node output layer 
model.add(layers.Dense(1))

# Create an optimiser that uses the Adam algorithm 
optimiser = Adam(learning_rate = 0.1)

# Compile the network with the optimiser, a loss metric, and an error metric
model = model.compile(loss='mse', metrics=['mae'], optimizer=optimiser)
