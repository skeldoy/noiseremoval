import os
import cv2

# Function to compare images
def image_diff(image1_path, image2_path, output_path):
    # Load images
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    
    # Check if images have the same dimensions
    if img1.shape != img2.shape:
        print(f"Error: Images {image1_path} and {image2_path} have different dimensions.")
        return
    
    # Compute absolute difference between images
    diff = cv2.absdiff(img1, img2)
    
    # Save difference image
    cv2.imwrite(output_path, diff)
    print(f"Difference image saved at: {output_path}")

# Directories
training_dir = "../data/training"
autoencoders_dir = "../data/n2n"
output_dir = "../data/n2n-diff"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Iterate through images in both directories
for filename in os.listdir(training_dir):
    training_image_path = os.path.join(training_dir, filename)
    autoencoder_image_path = os.path.join(autoencoders_dir, filename)
    output_image_path = os.path.join(output_dir, filename)
    
    # Check if corresponding image exists in autoencoders directory
    if os.path.isfile(autoencoder_image_path):
        # Compute difference and save
        image_diff(training_image_path, autoencoder_image_path, output_image_path)
    else:
        print(f"No corresponding image found for {filename} in autoencoders directory.")

