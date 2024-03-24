import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from featureExtractors import features, labels

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


