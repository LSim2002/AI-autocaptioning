# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 10:38:22 2023

@author: LSIMON
"""

from tensorflow.keras.models import load_model
from sklearn.impute import SimpleImputer
import numpy as np
import joblib
import argparse


from landmarkdetect import getAngles



def main(precision,impath):

    #precision = 0.86

    
    model = load_model(f'C:\\Users\\lsimon\\Desktop\\autocaptioning\\LandmarkToPose(NN)\\models\\PCM_{precision}.keras')
    imputer = joblib.load(f'C:\\Users\\lsimon\\Desktop\\autocaptioning\\LandmarkToPose(NN)\\models\\imputer_{precision}.pkl')
    
    
    
    #angles = getAngles('C:\\Users\\lsimon\\Desktop\\autocaptioning\\LandmarkToPose(NN)\\running2\\76.png')
    angles = getAngles(impath)
    angle_values = [angle[1] for angle in sorted(angles, key=lambda x: x[0])]
    
    
    if all(value == -1 for value in angle_values):
        print('aucun corps humain detecté')
        
    else:
            
        # Reshape the input to match the input shape for the model
        input_data = np.array(angle_values).reshape(1, -1)
        
        # Replace missing values
        input_data[input_data == -1] = np.nan
        
        # Transform the data with the imputer
        input_data = imputer.transform(input_data)
        
        
        
        
        
        prediction = model.predict(input_data)
        
        
        
        ##le model output avec 1 seul neurone. Si ce neurone vaut 0 alors cest une marche et 
        #si ce neurone vaut 1 alors cest running. 
        if prediction >= 0.5:
            #print("The model predicts this person is running.")
            print("The model is {:.2%} sure that the person is running.".format(prediction[0][0]))
        
        else:
            #print("The model predicts this person is walking.")0.86
            print("The model is {:.2%} sure that the person is walking.".format(1-prediction[0][0]))
            
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision")
    parser.add_argument("--impath")
    args = parser.parse_args()
    
    # If precision argument is not given, prompt the user to input it
    if args.precision is None:
        args.precision = float(input("Enter the precision: "))
    
    # If impath argument is not given, prompt the user to input it
    if args.impath is None:
        args.impath = input("Enter the image path: ")

    main(args.precision, args.impath)
    