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



def main(precision,angles):

    #precision = 0.86

    
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
        
        # Get the predicted class
        predicted_class = np.argmax(prediction)
        
        #print(f"The model predicts this person is {class_labels[predicted_class]}, with {np.max(prediction)*100:.2f}% confidence.")
        for i, class_label in enumerate(class_labels):
            print(f"The model's confidence that the person is {class_label} is {prediction[0][i]*100:.2f}%.")
            
    
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

    main(args.precision, getAngles(args.impath))
    