######################## SVM model ##########################
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file into a pandas dataframe
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

# Define the SVM model
svm = SVC()

# Define the hyperparameters to optimize using GridSearchCV
param_grid = {'C': [10**i for i in range(-5,5)],
              'kernel': ['linear', 'poly', 'rbf','sigmoid'],
              'degree': [2, 3, 4, 5],
              'gamma': ['auto', 'scale']}

# Define the GridSearchCV object and fit it to the training data
grid_search = GridSearchCV(svm, param_grid, cv=5, n_jobs=-1, verbose=3, refit=True)

# Fit the GridSearchCV object to the training data and get the results
history = grid_search.fit(X_train, y_train)

# Get the index of the best mean test score
best_idx = np.argmax(history.cv_results_['mean_test_score'])

# Print the best hyperparameters and the corresponding accuracy score
print("Best hyperparameters: ", grid_search.best_params_)
print("Accuracy score: ", grid_search.best_score_)
print("Iteration: ", best_idx + 1)

# Plot the number of iterations, accuracy and losses
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.cv_results_['rank_test_score'], history.cv_results_['mean_test_score'], 'o-', label='Test accuracy')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.subplot(1, 2, 2)
plt.plot(history.cv_results_['rank_test_score'], history.cv_results_['mean_fit_time'], 'o-', label='Fit time')
plt.plot(history.cv_results_['rank_test_score'], history.cv_results_['mean_score_time'], 'o-', label='Score time')
plt.xlabel('Iteration')
plt.ylabel('Time (s)')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# Predict the labels of the testing set using the optimized model
y_pred = grid_search.predict(X_test)

# Compute the accuracy score and confusion matrix on the testing set
accuracy = accuracy_score(y_test, y_pred)
conf_matrix_test = confusion_matrix(y_test, y_pred)

# Print the accuracy score and confusion matrix on the testing set
print("Accuracy score on the testing set: ", accuracy)
print("Confusion matrix on the testing set: ")
print(conf_matrix_test)

# Compute the accuracy score and confusion matrix on the training set
y_pred_train = grid_search.predict(X_train)
accuracy_train = accuracy_score(y_train, y_pred_train)
conf_matrix_train = confusion_matrix(y_train, y_pred_train)

# Print the accuracy score and confusion matrix on the training set
print("Accuracy score on the training set: ", accuracy_train)
print("Confusion matrix on the training set: ")
print(conf_matrix_train)

# Visualize the confusion matrix on the testing set using a heatmap
labels = df['Class'].unique()
sns.heatmap(conf_matrix_test, annot=True, fmt='g', xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Predicted label')
plt.ylabel('True label')
# plt.title('Confusion matrix on the testing set')
plt.tight_layout()
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