# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 16:01:25 2023

@author: LSIMON
"""



from tensorflow.keras.models import load_model
from sklearn.impute import SimpleImputer
import numpy as np
import joblib
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from landmarkdetect import getAngles

import tensorflow as tf

import logging
tf.get_logger().setLevel(logging.ERROR)

def getActions(precision,angles):

    #precision = 0.86
    if len(angles)==0: return None
    
    model = load_model(f'C:\\Users\\lsimon\\Desktop\\autocaptioning\\LandmarkToPose(NN)\\models\\PCM_{precision}_N.keras')
    imputer = joblib.load(f'C:\\Users\\lsimon\\Desktop\\autocaptioning\\LandmarkToPose(NN)\\models\\imputer_{precision}_N.pkl')
    
    
    
    #angles = getAngles('C:\\Users\\lsimon\\Desktop\\autocaptioning\\LandmarkToPose(NN)\\running2\\76.png')
    #angles = getAngles(impath)
    angle_values = [angle[1] for angle in sorted(angles, key=lambda x: x[0])]
    
    
    if all(value == -1 for value in angle_values):
        print('aucun corps humain detecté')
        
    else:
            
        # Reshape the input to match the input shape for the model
        input_data = np.array(angle_values).reshape(1, -1)
        
        
        # Transform the data with the imputer
        input_data = imputer.transform(input_data)
        
        
        prediction = model.predict(input_data)
        
        # Define your class labels (note: adjusted to reflect correct ordering)
        class_labels = ['running', 'walking', 'handshaking']
        
        res=[]
        #print(f"The model predicts this person is {class_labels[predicted_class]}, with {np.max(prediction)*100:.2f}% confidence.")
        i=0
        for class_label in class_labels:
            res.append([class_label, round( prediction[0][i]*100,2)])
            i+=1
            
    return res
            
    
    