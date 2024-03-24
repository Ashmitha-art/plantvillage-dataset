import os
import cv2
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import mahotas as mt

# extract color

def extract_color_histogram(image):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calculate the histogram for each channel
    blue_color = cv2.calcHist([hsv_image], [0], None, [256], [0, 256]) 
    red_color = cv2.calcHist([hsv_image], [1], None, [256], [0, 256]) 
    green_color = cv2.calcHist([hsv_image], [2], None, [256], [0, 256]) 
        
    # Concatenate histograms into a single feature vector
    hist_features = np.concatenate((blue_color, red_color, green_color), axis=None)
    
    return hist_features

# extract texture

def extractHaralick(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    textures = mt.features.haralick(image).mean(axis=0)
    return textures

# extract shape

def extractShape(image):
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image=image
    #calculate Moments
    moments = cv2.moments(gray_image)
    #calculate Hu Moments
    huMoments = cv2.HuMoments(moments)
    # convert huMoments to logarithmic scale for easier comparision 
    for i in range(7):
        huMoments[i] = -1 * np.copysign(1.0, huMoments[i]) * np.log10(abs(huMoments[i]))
    # Convert huMoments to row vector for eaiser storage with other extracted features
    huMoments = huMoments.flatten()
    return huMoments

# Path to the dataset folders
colored_folder = 'color'
grayscale_folder = 'grayscale'
segmented_folder = 'segmented'

# Initialize lists to store features and labels
features = []
labels = []

# Load images from each folder and extract color feature
for folder in [colored_folder, grayscale_folder, segmented_folder]:
    for subfolder in os.listdir(folder):
        if subfolder.startswith('Tomato') or subfolder.startswith('tomato'):
            folder_path = os.path.join(folder, subfolder)
            if os.path.isdir(folder_path):
                for inside_folder in os.listdir(folder_path):
                    image_path = os.path.join(folder, subfolder, inside_folder)
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"Failed to load image: {image_path}. Skipping...")
                        continue  # Skip the rest of the code in this loop iteration and proceed with the next image

                    # Extract color, shape, and texture for all images and append all folder names to labels(target classes)
                    hist_features = extract_color_histogram(image)
                    hu_moments = extractShape(image)
                    haralick_features = extractHaralick(image)
                    combined_features = np.concatenate((hist_features, hu_moments, haralick_features), axis=None)
                    features.append(combined_features)
                    labels.append(folder.split('/')[-1])  # Assuming folder structure is consistent

features = np.array(features)
labels = np.array(labels)






