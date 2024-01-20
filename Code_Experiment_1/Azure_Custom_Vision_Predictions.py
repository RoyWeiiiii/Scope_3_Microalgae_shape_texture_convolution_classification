from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry
from msrest.authentication import ApiKeyCredentials
import os, time
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

# Load the endpoint and keys of your resource
training_endpoint = ('https://southeastasia.api.cognitive.microsoft.com/')
training_key = ('0ac267da81354216b38b707aa9dcc333')
prediction_endpoint = ('https://southeastasia.api.cognitive.microsoft.com/')
prediction_key = ('0ac267da81354216b38b707aa9dcc333')
prediction_resource_id = ('/subscriptions/1eefc508-750d-47e1-aae2-10fab405f629/resourceGroups/MicroalgaeV1/providers/Microsoft.CognitiveServices/accounts/Microalgae1')

# Authenticate the client
credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(training_endpoint, credentials)
prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(prediction_endpoint, prediction_credentials)

# Get path to images folder
dirname = os.path.dirname(__file__)
images_folder = os.path.join(dirname, 'images/Test')

# Create variables for your project
publish_iteration_name = 'Microalgae_Batch_1_Python_Azure'
project_id = "f0843f52-d404-4898-8cbd-b916627d3c86"

# Create variables for your prediction resource
prediction_key = "0ac267da81354216b38b707aa9dcc333"
endpoint = "https://southeastasia.api.cognitive.microsoft.com/"
prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(endpoint, prediction_credentials)

# Define the path to test images folder
test_images_folder_path = os.path.join(os.path.dirname(__file__), "Images_Batch_1", "Test")
test_images_folder_path = "E:\CodingProjects\machine_learning\Azure_Custom_Vision_V3\Batch 1\preprocessing_batch_1_Azure\Images_Batch_1\Test/"

# Encode the tags
encoder = LabelEncoder()
encoder.fit(["Spirulina_Platensis", "Chlorella_FSP", "Chlamydomonas_Reinhardtii"])
encoded_labels = encoder.transform(["Spirulina_Platensis", "Chlorella_FSP", "Chlamydomonas_Reinhardtii"])
decoded_labels = encoder.inverse_transform(encoded_labels)

# Set threshold value for prediction probabilities
threshold = 0.5

# Test - Make a prediction
print("Testing the prediction endpoint with threshold value...")
true_labels = []
pred_labels = []
for subfolder in os.listdir(test_images_folder_path):
    subfolder_path = os.path.join(test_images_folder_path, subfolder)
    if os.path.isdir(subfolder_path):
        for test_image_filename in os.listdir(subfolder_path):
            if test_image_filename.endswith('.jpg'):
                image_path = os.path.join(subfolder_path, test_image_filename)
                with open(image_path, "rb") as image_contents:
                    results = predictor.classify_image_with_no_store(project_id, publish_iteration_name, image_contents.read(), threshold=threshold)

                    # Get the true label and the predicted label
                    true_label = subfolder
                    if len(results.predictions) > 0:
                        pred_label = results.predictions[0].tag_name
                    else:
                        pred_label = "Unknown"
                    true_labels.append(true_label)
                    pred_labels.append(pred_label)

                    # Display the results
                    print(f"Testing image {test_image_filename}...")
                    for prediction in results.predictions:
                        print(f"\t{prediction.tag_name}: {prediction.probability*100 :.2f}%")

# Encode the true and predicted labels
true_labels = encoder.transform(true_labels)
pred_labels = encoder.transform(pred_labels)

# Compute the confusion matrix
cm = confusion_matrix(true_labels, pred_labels)

# Plot the heatmap
fig, ax = plt.subplots()
im = ax.imshow(cm, cmap='Blues')

# Add labels to the x and y axes
classes = ["Chlamydomonas_Reinhardtii", "Chlorella_FSP", "Spirulina_Platensis"]
ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)

# Rotate the x-axis labels
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Add numbers to each cell in the confusion matrix
thresh = cm.max() / 2.
for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

plt.colorbar(im)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix Heatmap')
plt.show()

# Compute the overall accuracy
overall_accuracy = accuracy_score(true_labels, pred_labels)
print(f"Overall accuracy: {overall_accuracy*100:.2f}%")

# Compute precision, recall, and F1-score for each class
class_names = ["Chlamydomonas_Reinhardtii", "Chlorella_FSP", "Spirulina_Platensis"]
metrics = {}
for i, class_name in enumerate(class_names):
    tp = cm[i,i]
    fp = np.sum(cm[:,i]) - tp
    fn = np.sum(cm[i,:]) - tp
    tn = np.sum(cm) - tp - fp - fn
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    metrics[class_name] = {"Precision": precision, "Recall": recall, "F1-score": f1_score}
    
# Print the metrics
for class_name, metric in metrics.items():
    print(f"Metrics for class {class_name}:")
    for metric_name, metric_value in metric.items():
        print(f"\t{metric_name}: {metric_value:.2f}")