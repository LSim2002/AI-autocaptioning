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

    
       # Initialize directories
    directories = [
        "C:\\Users\\lsimon\\Desktop\\autocaptioning\\LandmarkToPose(NN)\\running2",  # Replace with your actual directories
        "C:\\Users\\lsimon\\Desktop\\autocaptioning\\LandmarkToPose(NN)\\walking2",
        "C:\\Users\\lsimon\\Desktop\\autocaptioning\\LandmarkToPose(NN)\\handshake2",
        # Add more directories as needed
    ]
    
    # Initialize your X and y
    X = []
    y = []
    
    # Get your data
    for i, directory in enumerate(directories):
        dir_name = os.path.basename(directory)  # get the base name of the directory
        for filename in tqdm(os.listdir(directory), desc=f"Processing {dir_name}"):
            if filename.endswith(".png"):  # assuming images are .png, modify as needed
                filepath = os.path.join(directory, filename)
    
                # Get the angles from your function
                angles = getAngles(filepath)  # Replace with your actual function
    
                # Convert angles to a list of values while preserving the order
                angle_values = [angle[1] for angle in sorted(angles, key=lambda x: x[0])]
    
                # Append to your data
                X.append(angle_values)
                y.append(i)  # i serves as the label here, incrementing for each directory
    
    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)
    

    # Save
    np.save('C:\\Users\\lsimon\\Desktop\\autocaptioning\\LandmarkToPose(NN)\\trainingdata\\XN.npy', X)
    np.save('C:\\Users\\lsimon\\Desktop\\autocaptioning\\LandmarkToPose(NN)\\trainingdata\\yN.npy', y)


if __name__ == "__main__":
    main()


##on utilise main pour que ce code ne sexecute pas si on importe le script dans un autre script






