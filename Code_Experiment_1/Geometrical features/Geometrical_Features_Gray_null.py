############################ Base case for geometrical features (Gray-scaling & Adaptive Thresholding) ###########################
import cv2
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Define a function to extract the morphological features
def extract_features(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Find the contours in the binary image
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize an empty list to store the features for each contour
    features_list = []

    # Loop over the contours and extract the features
    for cnt in contours:
        # Check if the contour has at least five points
        if len(cnt) < 5:
            continue

        # Calculate the area, perimeter, and Feret diameter of the contour
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        _, (w, h), _ = cv2.minAreaRect(cnt)
        feret_diameter = max(w, h)

        # Calculate the solidity of the contour
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            solidity = 0
        else:
            solidity = float(area) / hull_area

        # Calculate the extent of the contour
        rect = cv2.boundingRect(cnt)
        rect_area = rect[2] * rect[3]
        if rect_area == 0:
            extent = 0
        else:
            extent = float(area) / rect_area

        # Calculate the aspect ratio of the contour
        aspect_ratio = float(w) / h if w > h else float(h) / w

        # Calculate the circularity of the contour
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        # Calculate the eccentricity and convexity of the contour
        (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
        if ma == 0:
            eccentricity = 0
        else:
            eccentricity = np.sqrt(1 - (MA / ma) ** 2)
        convexity = float(area) / cv2.contourArea(hull)

        # Add the features to the list
        features_list.append([eccentricity, convexity, feret_diameter, area, perimeter, solidity, extent, aspect_ratio, circularity])

    # If there are no contours, return None
    if not features_list:
        return None

    # Otherwise, calculate the average of the features for all the contours
    features_avg = np.mean(features_list, axis=0)

    # Return the average features
    return features_avg.tolist()


# Define the path to the parent folder containing the subfolders
parent_path = "Input parant path"

# Define a list of the subfolder names
subfolder_names= ["Chlamydomonas_Reinhardtii", "Chlorella_FSP", "Spirulina_Platensis"]

# Initialize an empty list to store the feature data
feature_data = []

# Loop over the subfolders and extract features from images in each subfolder
for subfolder_name in subfolder_names:
    subfolder_path = os.path.join(parent_path, subfolder_name)
    
    # Loop over the images in the subfolder and extract the features
    for image_name in os.listdir(subfolder_path):
        # Load the image
        image_path = os.path.join(subfolder_path, image_name)
        image = cv2.imread(image_path)

        # Extract the features from the image
        features = extract_features(image)

        # If features were successfully extracted, add them to the feature data
        if features is not None:
            feature_data.append(features + [subfolder_name])

# Define the column names for the dataframe
column_names = ["Eccentricity", "Convexity", "FeretDiameter", "Area", "Perimeter", "Solidity", "Extent", "AspectRatio", "Circularity", "Class"]

# Create a pandas dataframe from the feature data
df = pd.DataFrame(feature_data, columns=column_names)

# Apply MinMaxScaler to normalize the features between 0 and 1
# scaler = MinMaxScaler()
# df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])

# Apply Z-score normalization to standardize the features
scaler = StandardScaler()
df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])

# Group the dataframe by class
grouped = df.groupby('Class')

# Fill NaN values in Eccentricity column with mean value of the same class
df['Eccentricity'] = grouped['Eccentricity'].apply(lambda x: x.interpolate())

# Take the mean of the forward and back rows to fill NaN values
df['Eccentricity'] = (df['Eccentricity'].ffill() + df['Eccentricity'].bfill()) / 2

# Fill remaining NaN values with median
df = df.fillna(df.median())

# Print the dataframe
print(df)

# Export the DataFrame to a CSV file
root_dir = "Input root directory"
export_path = os.path.join(root_dir, 'Input export path to save file as.csv')
df.to_csv(export_path, index=False)