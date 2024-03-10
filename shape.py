import os
import cv2
import numpy as np
import matplotlib.pyplot as plt 

def extractShape(image):
    #calculate Moments
    moments = cv2.moments(image)
    #calculate Hu Moments
    huMoments = cv2.HuMoments(moments)
    # convert huMoments to logarithmic scale for easier comparision 
    for i in range(7):
        huMoments[i] = -1 * np.copysign(1.0, huMoments[i]) * np.log10(abs(huMoments[i]))
    # Convert huMoments to row vector for eaiser storage with other extracted features
    huMoments = huMoments.flatten()
    return huMoments

grayscale_folder = 'grayscale'

# Iterate only through the grayscale folder
for subfolder in os.listdir(grayscale_folder):
    folder_path = os.path.join(grayscale_folder, subfolder)
    if os.path.isdir(folder_path):
        for image_file in os.listdir(folder_path):
            image_path = os.path.join(grayscale_folder, subfolder, image_file)
            image = cv2.imread(image_path)
            if image is not None:
                # Extract features
                hu_moments = extractShape(image)
                features.append(hu_moments)
                labels.append(subfolder)  # Using the subfolder name as the label
            else:
                print("Failed to load image:", image_path)
                    