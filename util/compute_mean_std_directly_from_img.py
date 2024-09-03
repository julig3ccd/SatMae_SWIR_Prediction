import os
import numpy as np
import rasterio
from tqdm import tqdm

# Directory where the images are stored
dataset_dir = "~/data/train"

# Initialize variables to store the sum and sum of squares for each channel
sum_channels = np.zeros(13, dtype=np.float64)
sum_squares_channels = np.zeros(13, dtype=np.float64)
num_pixels = 0

# Iterate over all images in the dataset
for image_name in tqdm(os.listdir(dataset_dir)):
    image_path = os.path.join(dataset_dir, image_name)
    
    # Open the image using rasterio
    with rasterio.open(image_path) as src:
        # Read the entire image as a 3D NumPy array (bands, rows, cols)
        image = src.read()
        
        # Ensure the image has the expected 13 channels
        if image.shape[0] != 13:
            print(f"Warning: {image_name} does not have 13 channels. Skipping...")
            continue
        
        # Reshape to (rows * cols, bands) to make summing easier
        image = image.reshape(13, -1)
        
        # Accumulate the sum and sum of squares
        sum_channels += image.sum(axis=1)
        sum_squares_channels += (image ** 2).sum(axis=1)
        
        # Update the total number of pixels processed
        num_pixels += image.shape[1]

# Compute the mean for each channel
mean_channels = sum_channels / num_pixels

# Compute the standard deviation for each channel
std_channels = np.sqrt(sum_squares_channels / num_pixels - mean_channels ** 2)

# Print the results
print("Mean per channel:", mean_channels)
print("Standard deviation per channel:", std_channels)
