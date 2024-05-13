# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 12:12:23 2023

@author: LSIMON
"""


import numpy as np
import os
from sklearn.impute import SimpleImputer

from tqdm import tqdm


from landmarkdetect import getAngles
    
    




def main():

    # Your data preparation here
    # X, y = ...
    
    # Initialize directories
    running_dir = "C:\\Users\\lsimon\\Desktop\\autocaptioning\\LandmarkToPose(NN)\\running2"  # Replace with your actual directory
    walking_dir = "C:\\Users\\lsimon\\Desktop\\autocaptioning\\LandmarkToPose(NN)\\walking2"  # Replace with your actual directory
    
    # Initialize your X and y
    X = []
    y = []
    
    ##1=running, 0=walking
    
    # Get your data
    for directory, label in zip([running_dir, walking_dir], [1, 0]):  # Assuming 1: running, 0: walking
        dir_name = os.path.basename(directory)  # get the base name of the directory
        for filename in tqdm(os.listdir(directory), desc=f"Processing {dir_name}"):
            if filename.endswith(".png"):  # assuming images are .jpg, modify as needed
                filepath = os.path.join(directory, filename)
                
                # Get the angles from your function
                angles = getAngles(filepath)  # Replace with your actual function
                
                # Convert angles to a list of values while preserving the order
                angle_values = [angle[1] for angle in sorted(angles, key=lambda x: x[0])]
                
                # Append to your data
                X.append(angle_values)
                y.append(label)
    
    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    
    
    # Impute missing values in X with mean value of the feature
    imp = SimpleImputer(missing_values=-1, strategy='mean')
    X = imp.fit_transform(X)
    
    # Replace -1 values (missing values) with numpy NaN
    X[X == -1] = np.nan
    
    
    
    
    # Save
    np.save('C:\\Users\\lsimon\\Desktop\\autocaptioning\\LandmarkToPose(NN)\\trainingdata\\X.npy', X)
    np.save('C:\\Users\\lsimon\\Desktop\\autocaptioning\\LandmarkToPose(NN)\\trainingdata\\y.npy', y)


if __name__ == "__main__":
    main()


##on utilise main pour que ce code ne sexecute pas si on importe le script dans un autre script






