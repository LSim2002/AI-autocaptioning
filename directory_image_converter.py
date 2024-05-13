# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 15:45:24 2023

@author: LSIMON
"""

import os
from PIL import Image
import glob
import argparse


#def convert_to_png(input_dir, output_dir):
def main(input_dir,output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Image counter
    img_counter = 1

    # Loop over each file in the input directory
    for filename in glob.glob(os.path.join(input_dir, "*")):
        # If the file is an image that PIL supports, open it
        try:
            with Image.open(filename) as im:
                # Save the image in PNG format in the output directory with the counter as the name
                im.save(os.path.join(output_dir, str(img_counter) + '.png'))
                img_counter += 1
                
        except IOError:
            print(f"Cannot convert {filename}")
            

# Use the function
#convert_to_png("C:\\Users\\lsimon\\Desktop\\autocaptioning\\LandmarkToPose(NN)\\walking", "C:\\Users\\lsimon\\Desktop\\autocaptioning\\LandmarkToPose(NN)\\walking2")
#C:\Users\lsimon\Desktop\autocaptioning\LandmarkToPose(NN)\walking

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    parser.add_argument("--output_dir")
    args = parser.parse_args()
    
    # If precision argument is not given, prompt the user to input it
    if args.input_dir is None:
        args.input_dir = input("Enter the input_dir: ")
    
    # If impath argument is not given, prompt the user to input it
    if args.output_dir is None:
        args.output_dir = input("Enter the output_dir: ")

    main(args.input_dir, args.output_dir)
    
    
    
    
    