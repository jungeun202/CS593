# Reinitialize the paths and imports for the process
import shutil
from sklearn.cluster import KMeans
import cv2
import numpy as np
import os

# Paths from the initial setup
train_folder_path = '/Users/junghwang/Downloads/Human Action Recognition/train'
output_dir = 'categorized_train'

# Function to extract simple color histogram features
def extract_color_histogram(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

# Extract features from all images in the train folder
features = []
valid_image_paths = []

train_folder_contents = os.listdir(train_folder_path)  # Get list of files in train folder

for img_file in train_folder_contents:
    img_path = os.path.join(train_folder_path, img_file)
    hist_features = extract_color_histogram(img_path)
    if hist_features is not None:
        features.append(hist_features)
        valid_image_paths.append(img_path)

# Convert features to a NumPy array
features = np.array(features)

# Perform clustering (e.g., KMeans with 5 clusters)
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(features)

# Organize images into cluster directories
os.makedirs(output_dir, exist_ok=True)

for cluster_id in range(num_clusters):
    cluster_dir = os.path.join(output_dir, f'action{cluster_id + 1}')
    os.makedirs(cluster_dir, exist_ok=True)

for img_path, cluster_id in zip(valid_image_paths, clusters):
    cluster_dir = os.path.join(output_dir, f'action{cluster_id + 1}')
    shutil.copy(img_path, cluster_dir)

# Zip the categorized folder
final_zip_path = 'categorized_train.zip'
shutil.make_archive(final_zip_path.replace('.zip', ''), 'zip', output_dir)
