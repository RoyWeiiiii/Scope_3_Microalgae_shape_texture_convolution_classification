# ########################### k-NN model ##########################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.inspection import DecisionBoundaryDisplay

# Load the CSV file
df = pd.read_csv("Input the file path to load CSV file", header=0)

# Split the dataset into training and testing sets
X = df.drop('Class', axis=1).values
y = df['Class'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Convert y_train and y_test to Pandas Series
y_train = pd.Series(y_train)
y_test = pd.Series(y_test)

# Print number of training and testing images
print("Number of training images: ", len(X_train))
print("Number of testing images: ", len(X_test))

# Print number of images for each class in training and testing sets
print("Number of images for each class in training set:\n", y_train.value_counts())
print("Number of images for each class in testing set:\n", y_test.value_counts())

# Define the parameter grid for GridSearchCV and RandomizedSearchCV
param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'p': [1, 2]
}

# Initialize a K-NN classifier
knn = KNeighborsClassifier()

# Perform grid search to find the best hyperparameters using 5-fold cross validation
grid_search = GridSearchCV(knn, param_grid, cv=5, n_jobs=-1, verbose=3, refit=True)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print('Best hyperparameters:', grid_search.best_params_)

# Initialize a K-NN classifier with the best hyperparameters
knn = KNeighborsClassifier(**grid_search.best_params_)

# Train the classifier on the training data
history = knn.fit(X_train, y_train)

# Evaluate the classifier on the testing data
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
# Set tick labels
classes = np.unique(y)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=classes, yticklabels=classes,
       ylabel='True label',
       xlabel='Predicted label')
# Rotate the tick labels and set their alignment
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
# Loop over data dimensions and create text annotations
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
fig.tight_layout()
plt.show()

# Visualize the effect of changing the number of neighbors on the accuracy
n_neighbors_range = range(1, 21)
accuracy_scores = []
for n_neighbors in n_neighbors_range:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

plt.plot(n_neighbors_range, accuracy_scores)
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()

# Advanced evaluation metrics
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
classification = classification_report(y_test, y_pred)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 score: ", f1)
print("Classification report: ")
print(classification)