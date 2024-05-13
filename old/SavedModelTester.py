# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 12:47:06 2023

@author: LSIMON
"""

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from tensorflow.keras.models import load_model

from termcolor import colored


# Load data 
X = np.load('C:\\Users\\lsimon\\Desktop\\autocaptioning\\LandmarkToPose(NN)\\trainingdata\\X.npy')
y = np.load('C:\\Users\\lsimon\\Desktop\\autocaptioning\\LandmarkToPose(NN)\\trainingdata\\y.npy')



# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Define the imputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Fit the imputer on the training data
imputer.fit(X_train)

# Transform the training and test data
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)


loaded_model = load_model('C:\\Users\\lsimon\\Desktop\\autocaptioning\\LandmarkToPose(NN)\\models\\PCM.keras')

###test the model
loss, accuracy = loaded_model.evaluate(X_test, y_test)

print()
print(colored(f'Test accuracy: {accuracy}','green'))



