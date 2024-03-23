import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import mahotas as mt


def extractHaralick(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    textures = mt.features.haralick(image).mean(axis=0)
    return textures


grayscale_folder = 'grayscale'

features = []
labels = []

for subfolder in os.listdir(grayscale_folder):
    if subfolder.startswith('Tomato') or subfolder.startswith('tomato'):
        folder_path = os.path.join(grayscale_folder, subfolder)
        if os.path.isdir(folder_path):
            for image_file in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    haralick_features = extractHaralick(image)
                    features.append(haralick_features)
                    labels.append(subfolder)
                else:
                    print("Failed to load image:", image_path)

features = np.array(features)
labels = np.array(labels)

# Scaling features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    features_scaled, labels, test_size=0.2, random_state=42)

# Initialize the logistic regression classifier
logreg = LogisticRegression(max_iter=10000)  # Increase max_iter if needed

# Fit the model to the training data
logreg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logreg.predict(X_test)

# Calculate and print the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(
    f"Accuracy of logistic regression model with Haralick features: {accuracy}")
