###################### GLCM features (Grayscaling) ######################
import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import greycomatrix, greycoprops
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Define the GLCM properties to extract
properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']

# Define the distances and angles to consider
distances = [1, 3, 5]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4] # Angle at 0, 45, 90, 135 

# Define the image size to resize to
size = (512, 512)

# Define the main folder containing the subfolders
main_folder = "Input Main Folder path"

# Define the subfolders containing the images
subfolders = ['Chlamydomonas_Reinhardtii', 'Chlorella_FSP', 'Spirulina_Platensis']

# Initialize empty lists for the features and labels
features = []
labels = []

# Loop through the subfolders and extract features from each image
for label, subfolder in enumerate(subfolders):
    folder_path = os.path.join(main_folder, subfolder)
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, size)
        glcm = greycomatrix(img, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
        feature = [greycoprops(glcm, prop).ravel()[0] for prop in properties]
        features.append(feature)
        labels.append(subfolder)

# Convert the features and labels to a pandas dataframe
df = pd.DataFrame(features, columns=properties)
df['label'] = labels

# Normalize the features using z-score normalization
scaler = StandardScaler()
df[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])

# Print the first few rows of the dataframe
print(df)

# Export the DataFrame to a CSV file
root_dir = "Input root directory"
export_path = os.path.join(root_dir, 'Input export path to save file as.csv')
df.to_csv(export_path, index=False)