# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2
import argparse

import os 



def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

def getlandmarks(impath):
    res = []
    
    image = mp.Image.create_from_file(impath)

    
    base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)
    detection_result = detector.detect(image)
    pose_landmarks_list = detection_result.pose_landmarks
    
    
    if len(pose_landmarks_list)==0 :
        print( "aucun corps humain détecté" )
        return [[k,-1,-1,-1,0] for k in range(33)]
 
    else:
        for idx in range(len(pose_landmarks_list[0])):
            landmark = pose_landmarks_list[0][idx]
            res.append([idx,landmark.x,landmark.y,landmark.z,landmark.visibility])
        return res

##ON n'affiche un point sur l'image des landmarks que si sa vraisemblance est suffisante.. 
##choix arbitraire de google




def cv2_imshow(im):
    cv2.imshow('image',im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return



def ImShow(impath):
    img = cv2.imread(impath)
    cv2_imshow(img)
    

#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#

def KeyPointsImShow(impath):
    
    # STEP 2: Create an PoseLandmarker object.
    base_options = python.BaseOptions(model_asset_path='C:\\Users\\lsimon\\Desktop\\autocaptioning\\landmark detection\\pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)
    
    # STEP 3: Load the input image.
    image = mp.Image.create_from_file(impath)
    
    # STEP 4: Detect pose landmarks from the input image.
    detection_result = detector.detect(image)
    
    
    
    
    
    cv2.namedWindow("landmarks", cv2.WINDOW_NORMAL)
    cv2.namedWindow("segmentation", cv2.WINDOW_NORMAL)

    
    if len(detection_result.pose_landmarks)==0 :
        print("aucun corps humain détecté")
        return False
    else:
        # STEP 5: Process the detection result. In this case, visualize it.
        annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
        cv2.imshow("landmarks", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        #cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    
        ##Visualize the pose segmentation mask.
        segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
        visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
        cv2.imshow("segmentation", visualized_mask)
        cv2.waitKey(0)
        #cv2_imshow(visualized_mask)
        return True

#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#


def getAnglesOriented(coordinates):##deprecated
    # Make sure you have the right keypoints
    if len(coordinates) != 33:
        raise ValueError('Expected a list with 33 keypoints')
    
    # Create a dictionary to easily access keypoints by their id
    keypoints_dict = {keypoint[0]: np.array(keypoint[1:4]) for keypoint in coordinates}

    # Define a function to calculate the angle between two vectors
    def calculate_angle(a, b, c):
        vector_a_b = a - b
        vector_c_b = c - b
    
        dot_product = np.dot(vector_a_b, vector_c_b)
        cross_product = np.cross(vector_a_b, vector_c_b)
    
        angle_radians = np.arctan2(np.linalg.norm(cross_product), dot_product)
    
        angle_degrees = np.degrees(angle_radians)
        
        if cross_product[2] < 0:  # If the rotation is clockwise, the angle is negative, so we adjust it
            angle_degrees = 360 - angle_degrees
    
        return angle_degrees
        

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





#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#
    
def getAngles(impath):
    
    
    def getlandmarks(impath):
        res = []
        
        image = mp.Image.create_from_file(impath)

        
        base_options = python.BaseOptions(model_asset_path='C:\\Users\\lsimon\\Desktop\\autocaptioning\\landmark detection\\pose_landmarker.task')
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True)
        detector = vision.PoseLandmarker.create_from_options(options)
        detection_result = detector.detect(image)
        pose_landmarks_list = detection_result.pose_landmarks
        
        
        if len(pose_landmarks_list)==0 :
            #print( "aucun corps humain détecté" )
            return [[k,-1,-1,-1,0] for k in range(33)]
     
        else:
            for idx in range(len(pose_landmarks_list[0])):
                landmark = pose_landmarks_list[0][idx]
                res.append([idx,landmark.x,landmark.y,landmark.z,landmark.visibility])
            return res
    
    
    
    
    
    
    
    coordinates=getlandmarks(impath)
    
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
    


#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#




def get_average_angles(directory): ##ne prend pas en compte les angles valant -1
    # Initialize a dictionary to store the sum of each angle and the count
    angle_sums = {}
    angle_counts = {}
    i=0
    # Iterate over each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".png"):  # assuming images are .jpg, modify as needed
            i+=1
            filepath = os.path.join(directory, filename)
            
            
            # Get the angles from the second function
            angles = getAngles(filepath)
            
            # Add each angle to the sum and increment the count
            for angle_name, angle_value in angles:
                # If the angle is -1, skip it
                if angle_value == -1:
                    continue

                if angle_name in angle_sums:
                    angle_sums[angle_name] += angle_value
                    angle_counts[angle_name] += 1
                else:
                    angle_sums[angle_name] = angle_value
                    angle_counts[angle_name] = 1

    # Calculate the average for each angle
    average_angles = [(angle_name, angle_sum / angle_counts[angle_name]) 
                      for angle_name, angle_sum in angle_sums.items()]
    print("moyenne sur ", i ," photos")
    return average_angles  


def main(impath):


    #impath = 'C:\\Users\\lsimon\\Desktop\\autocaptioning\\LandmarkToPose(NN)\\running\\UB6.jpg'
    
    
    #print(getlandmarks(impath))
    #print("run prof",getAngles(getlandmarks(impathrp)))
    #print(getMask(getlandmarks(impath)))
    #print("run face",getAngles(getlandmarks(impathrf)))
    #print("walk prof",getAngles(getlandmarks(impathwp)))
    #print("walk face",getAngles(getlandmarks(impathwf)))
    
    #print(getlandmarks(impathrf))
    
    #print(getAngles(impathrf))
    KeyPointsImShow(impath)
    
    
    ##print("AVERAGES FOR RUNNING",get_average_angles(impathr))
    ##print("AVERAGES FOR WALKING",get_average_angles(impathw))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--impath")
    args = parser.parse_args()
    
    # If precision argument is not given, prompt the user to input it
    if args.impath is None:
        args.impath = input("Enter the image path: ")
    


    main(args.impath)

































