##################################### Base case for Hu_moments features (Gray-scaling) #####################
import cv2
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Define function to extract flattened Hu moments from an image
def extract_hu_moments(img_path):
    # Read image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Find contours
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract flattened Hu moments for each contour
    hu_moments_list = []
    for contour in contours:
        # Calculate moments for the contour
        moments = cv2.moments(contour)
        # Calculate Hu moments for the moments
        hu_moments = cv2.HuMoments(moments)
        # Flatten the Hu moments array
        hu_moments_flat = hu_moments.flatten()
        # Append the flattened Hu moments to the list
        hu_moments_list.append(hu_moments_flat)

    # If no contours were found, return a list of zeros
    if len(hu_moments_list) == 0:
        return [0]*7
    
    # Calculate the mean Hu moments for all contours in the image
    mean_hu_moments = np.mean(hu_moments_list, axis=0)

    # Return the mean Hu moments for the image
    return mean_hu_moments.tolist()

# Define path to directory containing images
img_dir = "Input image directory"

# Initialize empty list to store extracted features
features_list = []

# Loop over subdirectories in the image directory
for sub_dir in os.listdir(img_dir):
    # Get full path to subdirectory
    sub_dir_path = os.path.join(img_dir, sub_dir)
    
    # Loop over images in subdirectory
    for img_name in os.listdir(sub_dir_path):
        # Get full path to image
        img_path = os.path.join(sub_dir_path, img_name)
        
        # Extract mean Hu moments from image
        hu_moments_mean = extract_hu_moments(img_path)
        
        # Append class label and mean Hu moments to features list
        features_list.append(hu_moments_mean + [sub_dir])

# Create pandas DataFrame from features list
columns = [f'Hu_{i}' for i in range(1, 8)] + ['class']
df = pd.DataFrame(features_list, columns=columns)

# Choose which scaler to use: 'minmax' for MinMaxScaler, 'standard' for StandardScaler
scaler_type = 'standard'

# Normalize the Hu moments using chosen scaler
hu_moments = df.iloc[:, :-1].values
if scaler_type == 'minmax':
    scaler = MinMaxScaler()
else:
    scaler = StandardScaler()
scaler.fit(hu_moments)
hu_moments_normalized = scaler.transform(hu_moments)

# Add the normalized Hu moments to the DataFrame
for i in range(7):
    df[f'Hu_{i+1}'] = hu_moments_normalized[:, i]

# Display the DataFrame
print(df)

# Export the DataFrame to a CSV file
root_dir = "Input root directory"
if scaler_type == 'minmax':
    export_path = os.path.join(root_dir, 'Input file name to save as .csv file')
else:
    export_path = os.path.join(root_dir, 'Input file name to save as .csv file')
df.to_csv(export_path, index=False)