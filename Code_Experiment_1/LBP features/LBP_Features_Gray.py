################################# LBP_Uniform_Features (Grayscaling) #######################
import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import MinMaxScaler, StandardScaler

path = "Input image path"
classes = ['Chlamydomonas_Reinhardtii', 'Chlorella_FSP', 'Spirulina_Platensis']

def extract_lbp_features(image, method='uniform', radius=3, n_points=8):
    lbp = local_binary_pattern(image, n_points, radius, method=method)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)

    return hist

features_list = []
labels_list = []
size = (512, 512)

for c in classes:
    folder_path = os.path.join(path, c)
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image_resize = cv2.resize(image, size)
            features = extract_lbp_features(image_resize, method='uniform') # 'default','ror','uniform','nri_uniform','var'
            features_list.append(features) 
            labels_list.append(c)

df = pd.DataFrame(features_list, columns=["feature_{}".format(i) for i in range(len(features_list[0]))])
df["label"] = labels_list

# Normalize the features using z-score normalization
scaler = StandardScaler()
df[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])

# Print the first few rows of the dataframe
print(df)

# Export the DataFrame to a CSV file
root_dir = "Input root directory"
export_path = os.path.join(root_dir, 'Input export path to save file as.csv')
df.to_csv(export_path, index=False)