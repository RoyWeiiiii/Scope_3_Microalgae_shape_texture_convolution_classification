################################# LBP_Uniform_Features (Grayscaling, Bilateral Blue, Adap Thresh & Sobel) #######################
import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import MinMaxScaler, StandardScaler

path = "Input image path"
classes = ['Chlamydomonas_Reinhardtii', 'Chlorella_FSP', 'Spirulina_Platensis']

def extract_lbp_features(image, method='uniform', radius=3, n_points=8):
    # Apply bilateral blur to smooth the image while preserving edges
    blurred = cv2.bilateralFilter(image, 40, 1, 25)
    # Apply adaptive thresholding to create a binary image
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Apply Sobel operator to the image
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    image_sobel_x = cv2.convertScaleAbs(sobelx)
    image_sobel_y = cv2.convertScaleAbs(sobely)
    # image_sobel_mag = cv2.magnitude(sobelx, sobely)
    image_sobel_bit = cv2.bitwise_or(sobelx, sobely)
    
    # Compute LBP features on the Sobel magnitude image
    lbp = local_binary_pattern(image_sobel_bit, n_points, radius, method=method)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)

    # Display the images
    cv2.imshow("Original", image)
    cv2.imshow("Sobel X", image_sobel_x)
    cv2.imshow("Sobel Y", image_sobel_y)
    cv2.imshow("Sobel Magnitude", image_sobel_bit)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
            features = extract_lbp_features(image_resize, method='uniform') # ['default','ror','uniform','nri_uniform','var']
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