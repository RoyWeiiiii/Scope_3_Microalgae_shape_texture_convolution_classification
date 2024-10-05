######################## SVM model ##########################
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file into a pandas dataframe
df = pd.read_csv("C:\Roy_PhD_Nott_work\PhD work\Research direction\Experiment 1\Submit_V1\Code_Experiment_1\Optimised_image_pre-processing_combined\Results_Batch 3\Combined_microalgae_Best_Optimised_3.csv", header=0)

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

# Plot the confusion matrix for the testing set
fig, ax = plt.subplots()
im = ax.imshow(conf_matrix_test, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)

# Set tick labels
classes = np.unique(y)
ax.set(xticks=np.arange(conf_matrix_test.shape[1]),
       yticks=np.arange(conf_matrix_test.shape[0]),
       xticklabels=classes, yticklabels=classes,
       ylabel='True label',
       xlabel='Predicted label')

# Rotate the tick labels and set their alignment
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Loop over data dimensions and create text annotations
thresh = conf_matrix_test.max() / 2.
for i in range(conf_matrix_test.shape[0]):
    for j in range(conf_matrix_test.shape[1]):
        ax.text(j, i, format(conf_matrix_test[i, j], 'd'),
                ha="center", va="center",
                color="white" if conf_matrix_test[i, j] > thresh else "black")

fig.tight_layout()
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
