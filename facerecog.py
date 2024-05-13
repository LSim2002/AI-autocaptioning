# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 11:31:21 2023

@author: LSIMON
"""

from keras_facenet import FaceNet
import numpy as np
import os
import argparse
import cv2




#embedder = FaceNet()



#detections = embedder.extract("C:\\Users\lsimon\Downloads\Chine.jpg", threshold=0.95)

#embeddings = embedder.embeddings(detections)

#print(len(detections)) renvoie 2

#print(detections[1])



def euclidean_distance(arr1, arr2):
    if arr1.shape != arr2.shape:
        raise ValueError("Embeddings have different dimensions")
    return np.linalg.norm(arr1-arr2)



def save_embedding(image_path):
    # This function saves the embedding of an image to a specified path

    # Initialize the FaceNet model
    embedder = FaceNet()

    # Extract the detections
    detections = embedder.extract(image_path, threshold=0.95)

    # Check if any face is detected
    if not detections:
        raise ValueError(f"No face detected in the image at {image_path}")

    # Get the first detected face's embedding
    embedding = detections[0]['embedding']

    # Specify the save path
    save_path = "C:\\Users\\lsimon\\Desktop\\autocaptioning\\facialrecog\\faces\\"
    name = os.path.splitext(os.path.basename(image_path))[0]
    

    # Save the embedding as a .npy file
    np.save(f"{save_path}\\{name}", embedding)
    

#save_embedding('C:\\Users\\lsimon\\Desktop\\autocaptioning\\facialrecog\\faces\\UssainBolt.jpg')



def compare_image_to_saved_embeddings(image_path):
    saved_embeddings_dir = 'C:\\Users\\lsimon\\Desktop\\autocaptioning\\facialrecog\\faces'

    
   # Initialize FaceNet
    embedder = FaceNet()


    # Get the detections for the input image
    detections = embedder.extract(image_path, threshold=0.95)
                                                                  
    # We'll just use the first detection's embedding for this example
    input_image_embedding = detections[0]['embedding']                         


    
    # Iterate over all .npy files in the directory
    for file in os.listdir(saved_embeddings_dir):
        if file.endswith(".npy"):
            # Load the embedding from the .npy file
            saved_embedding = np.load(os.path.join(saved_embeddings_dir, file))
            
            # Calculate the distance
            distance = euclidean_distance(input_image_embedding, saved_embedding)
            
            # Get the name of the person from the file name
            person_name = os.path.splitext(file)[0]
            
            # Print the distance
            print(f"Distance to {person_name}: {distance}")





#compare_image_to_saved_embeddings('C:\\Users\\lsimon\\Downloads\\fauxtrump.jpg')


def identify_faces(image_path):
    saved_embeddings_dir='C:\\Users\\lsimon\\Desktop\\autocaptioning\\facialrecog\\faces'
    
    
    # Initialize FaceNet
    embedder = FaceNet()

    # Get the detections for the input image
    detections = embedder.extract(image_path, threshold=0.95)

    # Load the image for drawing
    image = cv2.imread(image_path)

    # For each detected person in the image
    for detection in detections:
        # Get this person's embedding and box
        detected_embedding = detection['embedding']
        box = detection['box']

        min_distance = float('inf')
        best_match_name = None

        # Iterate over all .npy files in the directory
        for file in os.listdir(saved_embeddings_dir):
            if file.endswith(".npy"):
                # Load the embedding from the .npy file
                saved_embedding = np.load(os.path.join(saved_embeddings_dir, file))

                # Calculate the distance
                distance = euclidean_distance(detected_embedding, saved_embedding)

                # If this distance is smaller than our current smallest, update it
                if distance < min_distance:
                    min_distance = distance
                    best_match_name = os.path.splitext(file)[0]

        # Draw the box and the name on the image
        x, y, width, height = box
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(image, best_match_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the image
    cv2.imshow('Identified Faces', image)

    # Wait for any key to close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()


##identify_faces('C:\\Users\\lsimon\\Downloads\\trumpputin.jpg')
#identify_faces('C:\\Users\\lsimon\\Desktop\\autocaptioning\\LandmarkToPose(NN)\\TESTSAMPLES\\14Y.png')


def locate_and_identify_faces(image_path):
    saved_embeddings_dir = 'C:\\Users\\lsimon\\Desktop\\autocaptioning\\facialrecog\\faces'
    
    
    
    # Initialize FaceNet
    embedder = FaceNet()

    # Get the detections for the input image
    detections = embedder.extract(image_path, threshold=0.95)

    # Load the image for determining the position
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    identified_faces = []

    # For each detected person in the image
    for detection in detections:
        # Get this person's embedding and box
        detected_embedding = detection['embedding']
        box = detection['box']

        min_distance = float('inf')
        best_match_name = None

        # Iterate over all .npy files in the directory
        for file in os.listdir(saved_embeddings_dir):
            if file.endswith(".npy"):
                # Load the embedding from the .npy file
                saved_embedding = np.load(os.path.join(saved_embeddings_dir, file))

                # Calculate the distance
                distance = euclidean_distance(detected_embedding, saved_embedding)

                # If this distance is smaller than our current smallest, update it
                if distance < min_distance:
                    min_distance = distance
                    best_match_name = os.path.splitext(file)[0]

        # Determine the position
        x, y, _, _ = box
        if x < width / 3:
            x_position = 'left'
        elif x < 2 * width / 3:
            x_position = 'center'
        else:
            x_position = 'right'

        if y < height / 3:
            y_position = 'top'
        elif y < 2 * height / 3:
            y_position = 'middle'
        else:
            y_position = 'bottom'

        position = f'{y_position} {x_position}'

        identified_faces.append((best_match_name, position))

    return identified_faces



#identify_faces('C:\\Users\\lsimon\\Downloads\\fauxtrump.jpg')

#print(locate_and_identify_faces('C:\\Users\\lsimon\\Downloads\\cronma.jpg'))



















