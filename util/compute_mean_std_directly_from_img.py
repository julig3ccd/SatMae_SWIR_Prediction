import os
import numpy as np
import rasterio
from tqdm import tqdm

# Directory where the images are stored
dataset_dir = "/home/stud/geissinger/data/validate"

# Initialize variables to store mean and M2 (sum of squares of differences from the mean)
mean_channels = np.zeros(13, dtype=np.float64)
M2_channels = np.zeros(13, dtype=np.float64)
num_pixels = 0

dir_list = os.listdir(dataset_dir)
dir_list = dir_list[:100]
# Iterate over all images in the dataset
for image_name in tqdm(dir_list):
    image_path = os.path.join(dataset_dir, image_name)
    
    # Open the image using rasterio
    with rasterio.open(image_path) as src:
        # Read the entire image as a 3D NumPy array (bands, rows, cols)
        image = src.read()
        
        # Ensure the image has the expected 13 channels
        if image.shape[0] != 13:
            print(f"Warning: {image_name} does not have 13 channels. Skipping...")
            continue
        
        # Reshape to (bands, rows * cols) to facilitate processing
        image = image.reshape(13, -1)
        
        # Incrementally calculate mean and variance using Welford's method
        for i in range(image.shape[1]):  # Iterate over pixels
            num_pixels += 1
            delta = image[:, i] - mean_channels
            mean_channels += delta / num_pixels
            delta2 = image[:, i] - mean_channels
            M2_channels += delta * delta2

# Calculate variance and standard deviation
variance_channels = M2_channels / num_pixels
std_channels = np.sqrt(variance_channels)

# Print the results
print("Mean per channel:", mean_channels)
print("Variance per channel:", variance_channels)
print("Standard deviation per channel:", std_channels)
