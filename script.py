import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow	import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import r2_score

'''
The aim of this neural network is to predict a student's probability of acceptance to a university based on the following features:
Serial No.,GRE Score, TOEFL Score, University Rating, SOP, LOR ,CGPA, Research
'''

# Load in the data as a dataframe 
dataset = pd.read_csv("admissions_data.csv")

print(dataset.columns)