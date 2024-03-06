import os
import cv2
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Function to extract color histogram features from images
def extract_color_histogram(image):
    print("hello")
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calculate the histogram for each channel
    blue_color = cv2.calcHist([hsv_image], [0], None, [256], [0, 256]) 
    red_color = cv2.calcHist([hsv_image], [1], None, [256], [0, 256]) 
    green_color = cv2.calcHist([hsv_image], [2], None, [256], [0, 256]) 
        
    # Concatenate histograms into a single feature vector
    hist_features = np.concatenate((blue_color, red_color, green_color), axis=None)
    
    return hist_features

# Path to the dataset folders
colored_folder = 'color'
grayscale_folder = 'grayscale'
segmented_folder = 'segmented'

# Initialize lists to store features and labels
features = []
labels = []
  
# Load images from each folder and extract features
for folder in [colored_folder, grayscale_folder, segmented_folder]:
    for subfolder in os.listdir(folder):
        folder_path = os.path.join(folder, subfolder)
        if os.path.isdir(folder_path):
            for inside_folder in os.listdir(folder_path):
                image_path = os.path.join(folder, subfolder, inside_folder)
                image = cv2.imread(image_path)
                if image is not None:
                    # Extract features
                    hist_features = extract_color_histogram(image)
                    features.append(hist_features)
                    labels.append(folder.split('/')[-1])  # Assuming folder structure is consistent
                else:
                    print("Failed to load image:", image_path)
# Convert lists to numpy arrays
features = np.array(features)
labels = np.array(labels)
print("hello")

# Split the data into training and testing sets (90-10 split)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42)

# Initialize the random forest classifier
rf_classifier = RandomForestClassifier()

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = rf_classifier.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("hello")