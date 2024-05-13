# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 13:49:54 2023

@author: LSIMON
"""
import os 




import numpy as np
from keras_facenet import FaceNet

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


from PIL import Image


from testModelNangles import getActions
from facerecog import identify_faces
import argparse

import openai

import logging






def getActionsAndNames(precision,impath):
    
    
    
    
##################obtenir les landmarks de chaque humain####################### 
    
    
    
    def getLandmarksLists(impath):
        # STEP 2: Create an PoseLandmarker object.
        base_options = python.BaseOptions(model_asset_path='C:\\Users\\lsimon\\Desktop\\autocaptioning\\landmark detection\\pose_landmarker.task')
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True,
            num_poses=4)
        detector = vision.PoseLandmarker.create_from_options(options)
        
        # STEP 3: Load the input image.
        image = mp.Image.create_from_file(impath)
        
        # STEP 4: Detect pose landmarks from the input image.
        detection_result = detector.detect(image)
        pose_landmarks_lists = detection_result.pose_landmarks



        if len(pose_landmarks_lists)==0 :
            print("aucun corps humain détecté!")
            return 0

        else:
            #print(len(pose_landmarks_lists))
            #return detection_result.pose_landmarks
            res=[]
            for idpers in range(len(pose_landmarks_lists)):
                tempres = []
                for idx in range(len(pose_landmarks_lists[idpers])):
                    landmark = pose_landmarks_lists[idpers][idx]
                    tempres.append([idx,landmark.x,landmark.y,landmark.z,landmark.visibility])
                res.append(tempres)
            return res
        
        
    print('#------detecting landmarks------#')
    LandmarksLists = getLandmarksLists(impath)
    print('#------done------#')

    if LandmarksLists==0:
        return 'fin du programme'
    #print(LandmarksLists)
    
##########obtenir les visages et les boxes correspondates######################
    
    def locate_and_identify_faces(image_path):
        def euclidean_distance(arr1, arr2):
            if arr1.shape != arr2.shape:
                raise ValueError("Embeddings have different dimensions")
            return np.linalg.norm(arr1-arr2)
        
        saved_embeddings_dir = 'C:\\Users\\lsimon\\Desktop\\autocaptioning\\facialrecog\\faces'
        
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        
        # Initialize FaceNet
        embedder = FaceNet()
    
        # Get the detections for the input image
        detections = embedder.extract(image_path, threshold=0.95)
    
        identified_faces = []
    
        # For each detected person in the image
        for detection in detections:
            min_dist = float('inf')
            closest_person = None
    
            # Embedding for the current face
            face_embedding = detection['embedding']
            face_box = detection['box']
    
            # Iterate over all .npy files (saved embeddings)
            for npy_file in os.listdir(saved_embeddings_dir):
                if npy_file.endswith('.npy'):  # only load .npy files
                    # Load saved embedding
                    saved_embedding = np.load(os.path.join(saved_embeddings_dir, npy_file), allow_pickle=True)
        
                    # Compute euclidean distance
                    dist = euclidean_distance(face_embedding, saved_embedding)
        
                    # Update closest person if this distance is smaller
                    if dist < min_dist:
                        min_dist = dist
                        closest_person = os.path.splitext(npy_file)[0]
    
            # Normalize the box coordinates and dimensions
            x, y, width, height = face_box
            x_normalized = x / img_width
            y_normalized = y / img_height
            width_normalized = width / img_width
            height_normalized = height / img_height
            
            # Calculate face center
            center_x = x_normalized + width_normalized / 2
            center_y = y_normalized + height_normalized / 2
            
            # Define face location based on center
            if center_y < 1/3:
                vertical_position = 'top'
            elif center_y < 2/3:
                vertical_position = 'center'
            else:
                vertical_position = 'bottom'
            
            if center_x < 1/3:
                horizontal_position = 'left'
            elif center_x < 2/3:
                horizontal_position = 'center'
            else:
                horizontal_position = 'right'
            
            face_location = f'{vertical_position}-{horizontal_position}'
    
            normalized_box = [x_normalized, y_normalized, width_normalized, height_normalized]
    
            # Append the identified face (name and box coordinates)
            identified_faces.append([closest_person, face_location, normalized_box])    
            
        return identified_faces #The bounding box is represented by a list of 
        #four numbers [x, y, width, height], where x and y are the coordinates
        #of the top-left corner, and width and height are the dimensions of 
        #the box.
        
    
    print('#------detecting faces------#')
    NamesAndBoxes=locate_and_identify_faces(impath)
    print('#------done------#')

    #print(NamesAndBoxes)
    
    
    

    
    #regarder les landmarks de tete de chaque personnes. Si ils sont dans la box
    ##d'une personne alors on peut la labeliser
    
    for a in range(len(NamesAndBoxes)):
        box_x, box_y, box_width, box_height = NamesAndBoxes[a][2]
        maxcount=0
        idpersmax=0
        for idpers in range(len(LandmarksLists)):
            landmarks=LandmarksLists[idpers]
            count = 0
            
            for i in range(0,11):
                landmark_id, x, y, _, visibility_score = landmarks[i]
                if visibility_score > 0.4 and box_x <= x <= (box_x + box_width) and box_y <= y <= (box_y + box_height):
                    count += 1
                    
            if count >= maxcount:
                maxcount=count
                idpersmax=idpers
        
        NamesAndBoxes[a][2] = idpersmax
            
    #print(NamesAndBoxes)
    ##[['poutine', 'top-right', 1], ['trump', 'top-left', 0]] ici poutine est le deuxieme set de landmarks
    ##trump est le premier set de landmarks
    
            
            
    
    
    
    
    #obtenir les angles de chaque humain
    
    def getAngles(coordinates):
        
        
        
        # Make sure you have the right keypoints
        if len(coordinates) != 33:
            raise ValueError('Expected a list with 33 keypoints')
        
        # Create a dictionary to easily access keypoints by their id
        keypoints_dict = {keypoint[0]: np.array(keypoint[1:5]) for keypoint in coordinates}

        # Define a function to calculate the angle between two vectors
        def calculate_angle(a, b, c):
            # Check visibility of keypoints
            if a[3] < 0.4 or b[3] < 0.4 or c[3] < 0.4:
                return -1.0
            
            vector_a_b = b[:3] - a[:3]
            vector_b_c = c[:3] - b[:3]

            dot_product = np.dot(vector_a_b, vector_b_c)

            magnitude_a_b = np.linalg.norm(vector_a_b)
            magnitude_b_c = np.linalg.norm(vector_b_c)

            angle_radians = np.arccos(dot_product / (magnitude_a_b * magnitude_b_c))

            return np.degrees(angle_radians)

        # Compute the angles for the left and right knee
        left_knee_angle = calculate_angle(keypoints_dict[23], keypoints_dict[25], keypoints_dict[27])
        right_knee_angle = calculate_angle(keypoints_dict[24], keypoints_dict[26], keypoints_dict[28])
        # Compute the angles for the left and right elbow
        left_elbow_angle = calculate_angle(keypoints_dict[11], keypoints_dict[13], keypoints_dict[15])
        right_elbow_angle = calculate_angle(keypoints_dict[12], keypoints_dict[14], keypoints_dict[16])
        # Compute the angles for the left and right hip
        left_hip_angle = calculate_angle(keypoints_dict[11], keypoints_dict[23], keypoints_dict[25])
        right_hip_angle = calculate_angle(keypoints_dict[11], keypoints_dict[24], keypoints_dict[26])
        # Compute the angles for the left and right shoulder
        left_shoulder_angle = calculate_angle(keypoints_dict[11], keypoints_dict[13], keypoints_dict[23])
        right_shoulder_angle = calculate_angle(keypoints_dict[11], keypoints_dict[14], keypoints_dict[24])
        # Compute the angles for the left and right ankle
        left_ankle_angle = calculate_angle(keypoints_dict[25], keypoints_dict[27], keypoints_dict[31])
        right_ankle_angle = calculate_angle(keypoints_dict[26], keypoints_dict[28], keypoints_dict[32])

        return [('left knee angle', round(left_knee_angle, 1)), ('right knee angle', round(right_knee_angle, 1)), 
            ('left elbow angle', round(left_elbow_angle, 1)), ('right elbow angle', round(right_elbow_angle, 1)),
            ('left hip angle', round(left_hip_angle, 1)), ('right hip angle', round(right_hip_angle, 1)),
            ('left shoulder angle', round(left_shoulder_angle, 1)), ('right shoulder angle', round(right_shoulder_angle, 1)),
            ('left ankle angle', round(left_ankle_angle, 1)), ('right ankle angle', round(right_ankle_angle, 1))]    
    

    
    
    
    #obtenir les actions de chaque humain
    
    print('#------getting angles and actions------#')

    for index in range(len(LandmarksLists)):
        curAngles= getAngles(LandmarksLists[index])
        actions = getActions(precision,curAngles)
        LandmarksLists[index]=actions
    
    print('#------done------#')


    #print(LandmarksLists)
    #faire correspondre les id
        
    for persondata in NamesAndBoxes:
        persondata[2] = LandmarksLists[persondata[2]]
    
    return NamesAndBoxes
        
        
        
        
        
        
        
        
        
    
impath = 'C:\\Users\\lsimon\\Desktop\\autocaptioning\\LandmarkToPose(NN)\\TESTSAMPLES\\27Y.png'
    
#print(getActionsAndNames(0.9,'C:\\Users\\lsimon\\Downloads\\piano.jpg'))
def text(liste):
    for person in liste:
        text = 'the image shows '+person[0]+' on the '+person[1]+'. There is '
        for action in person[2]:
            text += f"a {action[1]}" + "% chance that he is " + f"{action[0]} " 
       
        print(text)




#text(getActionsAndNames(0.9,impath))
       
#identify_faces(impath)
     


#model_lst = openai.Model.list()

#for i in model_lst['data']:
#    print(i['id'])
    
    

#print(getActionsAndNames(0.9,impath))
#liste = [['poutine', 'top-left', [['running', 13.73], ['walking', 3.11], ['handshaking', 83.16]]], ['obama', 'top-center', [['running', 6.91], ['walking', 8.28], ['handshaking', 84.81]]]]


    
    
def getPersonsActions(liste):
    res = ""
    for i in range(len(liste)):
        
        curactions = liste[i][2]
        text=liste[i][0]+' on the '+liste[i][1]+' of the picture, ' + max(curactions, key=lambda x: x[1])[0]+'. '

            
        res+=text
        if i != len(liste)-1:
            res+="The image also shows "    
    
    
    
    return res
        








#PersonsActions = getPersonsActions(getActionsAndNames(0.9,impath))


#Date = "November 30th, 2015"
#Place = "Paris"


#gptRequest = "Hello ChatGPT, when was Emmanuel Macron born?"
#gptRequest = "I have an image of Michael Phelps Swimming, August 10th, 2016, in Rio de Janeiro. Please give me a short caption for that image. 1 sentence, 35 words max, only factual, without opinion and judgement, and add some factual context (if you can guess some with what I gave you) about the image, which must be precisely about the image and not too general. Do not include the location and the date in the caption you generate."
#gptRequest = f'I have an image that shows {PersonsActions} The image was taken on {Date}, in {Place}. Please give me a short journalistic caption for that image. 1 sentence, 35 words max, only factual, without opinion and judgement, and add some RELEVANT factual context (knowing what event was happening there at that time which included the people I mentioned) about the image, which must be precisely about the image and not too general. Do not include the location and the date in the caption you generate. Please decide which action is most likely to be the correct one depicted, knowing the confidence of my pose detection model as well as the context. Do not include this "uncertainity" in your response.'

#print(gptRequest)























#openai.api_key =
#removed from github push for privacy
def getGPT(request,n):
    responses = []
    for _ in range(n):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": request},
            ]
        )
        responses.append(response['choices'][0]['message']['content'])

    # Join the responses with line breaks
    return '\n'.join(responses)


#print(getGPT(gptRequest))























def main(precision,impath,date,place,n):
    Date = date
    Place = place
    
    PersonsActions = getPersonsActions(getActionsAndNames(precision,impath))
    
    gptRequest = f'I have an image that shows {PersonsActions} The image was taken on {Date}, in {Place}. Please give me a short journalistic caption for that image. 1 sentence, 35 words max, only factual, without opinion and judgement, and add some RELEVANT factual context (knowing the NAME of the event that was happening there at that time which included the people I mentioned) about the image, which must be precisely about the image and not too general. Do not include the location and the date I gave you in the caption you generate.'
    
    print('#------prompting chat GPT------#')
    res = getGPT(gptRequest,n)
    print('#------done------#')

    print(res)
    
    return res



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", type=float)
    parser.add_argument("--impath")  # this argument is now required
    parser.add_argument("--date")  # this argument is now required
    parser.add_argument("--place")  # this argument is now required
    parser.add_argument("--numchoices", type=int)

    args = parser.parse_args()
    
    if args.precision is None:
        args.precision = float(input("Enter the precision (default 0.9): ") or 0.9)
    if args.impath is None:
        args.impath = input("Enter the image path: ")  # No default value
    if args.date is None:
        args.date = input("Enter the date: ")  # No default value
    if args.place is None:
        args.place = input("Enter the place: ")  # No default value
    if args.numchoices is None:
        args.numchoices = int(input("Enter the number of choices (default 3): ") or 3)

    main(args.precision, args.impath, args.date, args.place, args.numchoices)

    #print("hello")



#'C:\\Users\\lsimon\\Desktop\\autocaptioning\\LandmarkToPose(NN)\\TESTSAMPLES\\27Y.png'









        