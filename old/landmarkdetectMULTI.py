# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 13:16:12 2023

@author: LSIMON
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


def KeyPointsImShow(impath):
    
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
    
    
    
    #print(len(detection_result.pose_landmarks))
    
    cv2.namedWindow("landmarks", cv2.WINDOW_NORMAL)

    
    if len(detection_result.pose_landmarks)==0 :
        print("aucun corps humain détecté")
        return False
    else:
        # STEP 5: Process the detection result. In this case, visualize it.
        annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
        cv2.imshow("landmarks", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        #cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        #cv2_imshow(visualized_mask)
        return True
    
    
    
    
KeyPointsImShow('C:\\Users\\lsimon\\Downloads\\trumpputin.jpg')


