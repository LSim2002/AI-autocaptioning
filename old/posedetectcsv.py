# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 15:58:37 2023

@author: LSIMON
"""

import pandas as pd
from collections import Counter

##create dataframe
df = pd.read_csv('C:\\Users\\lsimon\\Desktop\\autocaptioning\\LandmarkToPose(NN)\\mpii_human_pose.csv')



##names = df['NAME']
##print(len(names))

##print(df.shape)
##renvoit (17372, 37) DONC il y a 17373 lignes dans le tabl complet (17372 lignes + 1 ligne pour name, id,rankle,etc)
##et 37 attributs y comprit name etc. soit: 37-5 = 32 coordonnées soit (en 2D) 16 keypoints


########################valeur de coordonnée vaut -1 si INVISIBLE###########################

#ressort,cf mediapipe, [11,12,13,14,15,16,23,24,25,26,27,28]
def getCoords(idx):
    rankle_X = df.loc[idx, 'r ankle_X']
    rankle_Y = df.loc[idx, 'r ankle_Y']
    lankle_X = df.loc[idx, 'l ankle_X']
    lankle_Y = df.loc[idx, 'l ankle_Y']
    rankle_coords=(rankle_X,rankle_Y)
    lankle_coords=(lankle_X,lankle_Y)

    rknee_X = df.loc[idx, 'r knee_X']
    rknee_Y = df.loc[idx, 'r knee_Y']
    lknee_X = df.loc[idx, 'l knee_X']
    lknee_Y = df.loc[idx, 'l knee_Y']
    rknee_coords = (rknee_X, rknee_Y)
    lknee_coords = (lknee_X, lknee_Y)


    rhip_X = df.loc[idx, 'r hip_X']
    rhip_Y = df.loc[idx, 'r hip_Y']
    lhip_X = df.loc[idx, 'l hip_X']
    lhip_Y = df.loc[idx, 'l hip_Y']
    rhip_coords = (rhip_X, rhip_Y)
    lhip_coords = (lhip_X, lhip_Y)

    #pelvis_X = df.loc[idx, 'pelvis_X']
    #pelvis_Y = df.loc[idx, 'pelvis_Y']

    #thorax_X = df.loc[idx, 'thorax_X']
    #thorax_Y = df.loc[idx, 'thorax_Y']

    #upperneck_X = df.loc[idx, 'upper neck_X']
    #upperneck_Y = df.loc[idx, 'upper neck_Y']

    #headtop_X = df.loc[idx, 'head top_X']
    #headtop_Y = df.loc[idx, 'head top_Y']

    rwrist_X = df.loc[idx, 'r wrist_X']
    rwrist_Y = df.loc[idx, 'r wrist_Y']
    lwrist_X = df.loc[idx, 'l wrist_X']
    lwrist_Y = df.loc[idx, 'l wrist_Y']
    rwrist_coords = (rwrist_X, rwrist_Y)
    lwrist_coords = (lwrist_X, lwrist_Y)

    relbow_X = df.loc[idx, 'r elbow_X']
    relbow_Y = df.loc[idx, 'r elbow_Y']
    lelbow_X = df.loc[idx, 'l elbow_X']
    lelbow_Y = df.loc[idx, 'l elbow_Y']
    relbow_coords = (relbow_X, relbow_Y)
    lelbow_coords = (lelbow_X, lelbow_Y)

    rshoulder_X = df.loc[idx, 'r shoulder_X']
    rshoulder_Y = df.loc[idx, 'r shoulder_Y']
    lshoulder_X = df.loc[idx, 'l shoulder_X']
    lshoulder_Y = df.loc[idx, 'l shoulder_Y']
    rshoulder_coords = (rshoulder_X, rshoulder_Y)
    lshoulder_coords = (lshoulder_X, lshoulder_Y)
    
    
    
    

    res = [lshoulder_coords,rshoulder_coords,lelbow_coords,relbow_coords,lwrist_coords,rwrist_coords,lhip_coords,rhip_coords,lknee_coords,rknee_coords,lankle_coords,rankle_coords]
    return res

def getMask(idx):
    mask=[]
    coords = getCoords(idx)
    for coordXY in coords:
        if coordXY[0]==-1 or coordXY[1]==-1:
            mask.append(0)
        else:
            mask.append(1)
    return mask
    


def getActivity(idx):
     return df.loc[idx, 'Activity']
 
    
 
    
def showAllActivities():
    liste = []
    for idx in range(len(df['NAME'])):
        string = df.loc[idx,'Category']
        liste.append(string)
    counter = Counter(liste)
    result = sorted(list(counter.items()), key=lambda x: x[1], reverse=True)
    return result    


print(showAllActivities())

print(len(showAllActivities()))