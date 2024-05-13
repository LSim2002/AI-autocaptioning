# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 11:20:55 2023

@author: LSIMON
"""

import scipy.io


# Load the .mat file.
mat = scipy.io.loadmat('C:\\Users\\lsimon\\Desktop\\autocaptioning\\LandmarkToPose(NN)\\mpii_human_pose_v1_u12_2\pose_annotations.mat')

# Access the top-level 'RELEASE' field.
data = mat['RELEASE']

print(data.dtype.names)
annotation_mpii = mat.__dict__['annolist'][0, 2]
print(annotation_mpii)







'''
third_im_annot = image_annotations[3]

third_image_name = third_im_annot['image.name']
print(third_image_name)

'''




'''

# Access the annotations for a specific image index.
imgidx = 0  # replace with your actual image index
image_annotations = RELEASE['annolist'][0, imgidx]

# Access the body annotations for a specific person.
ridx = 0  # replace with your actual person index
person_annotations = image_annotations['annorect'][0, ridx]


print(person_annotations)






# Access the body joint annotations.
joint_annotations = person_annotations['annopoints']['point'][0]

# Create lists to hold the joint coordinates and visibility information.
joint_coordinates = []
joint_visibility = []

# Iterate over each joint annotation.
for joint in joint_annotations:
    # Access the coordinates of the joint.
    x = joint['x'][0, 0]
    y = joint['y'][0, 0]
    joint_coordinates.append((x, y))

    # Access the joint visibility.
    is_visible = joint['is_visible'][0, 0]
    joint_visibility.append(is_visible)


print(joint_coordinates)



#annots=data[0][0][0][0]
#print(len(annots))


#print(annots[250])


'''

