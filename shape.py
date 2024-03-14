import os
import cv2
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


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

features = []
labels = []

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

features = np.array(features)
labels = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize the decision tree classifier
DTC = DecisionTreeClassifier()

# Fit the model to the training data
DTC.fit(X_train, y_train)

# Make predictions on the test set
y_pred = DTC.predict(X_test)

# Calculate and print the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of descion tree model: {accuracy}")

knn = KNeighborsClassifier(n_neighbors=5)

# Fit the model to the training data
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Calculate and print the accuracy of the model
accuracy1 = accuracy_score(y_test, y_pred)
print(f"Accuracy of knn model: {accuracy1}")


