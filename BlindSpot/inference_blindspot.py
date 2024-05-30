
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.io import read_image, write_jpeg
import os
from glob import glob
import numpy as np

# Define the CNN architecture for BlindSpot (modified for denoising)
class BlindSpotDenoising(nn.Module):
    def __init__(self):
        super(BlindSpotDenoising, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(F.interpolate(x, scale_factor=2)))
        x = F.relu(self.conv5(F.interpolate(x, scale_factor=2)))
        x = self.conv6(F.interpolate(x, scale_factor=2))
        return x

# Function to load the latest trained model
def load_latest_model(model, device, directory='./'):
    model_files = glob(os.path.join(directory, 'denoising_model_epoch*.pth'))
    if not model_files:
        raise FileNotFoundError("No model files found in the specified directory.")
    latest_model_path = max(model_files, key=os.path.getctime)  # Get the most recently created file
    print(f"Loading model from {latest_model_path}")
    model.load_state_dict(torch.load(latest_model_path, map_location=device))
    model.to(device)
    return model

# Function to denoise an image using the trained model
def denoise_image(model, device, image):
    model.eval()
    with torch.no_grad():
        image = image.to(device).unsqueeze(0)  # Add batch dimension
        output = model(image)
        return output.squeeze(0)  # Remove batch dimension

# Create directory if it doesn't exist
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the BlindSpotDenoising model
model = BlindSpotDenoising()

# Load the latest trained model
model = load_latest_model(model, device, directory='./')

# Define a transformation for the dataset
transform = transforms.Compose([
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Paths for the noisy and denoised images
noisy_image_dir = '../data/training'
denoised_image_dir = '../data/blindspot'
create_dir(denoised_image_dir)

# Load and denoise images
noisy_image_paths = glob(os.path.join(noisy_image_dir, '*.png'))
for img_path in noisy_image_paths:
    noisy_image = read_image(img_path).float() / 255.0
    noisy_image = transform(noisy_image)
    denoised_image = denoise_image(model, device, noisy_image)

    # Convert the denoised image to the format required by write_jpeg
    denoised_image = denoised_image.cpu().clamp(0, 1)  # Clamp values to [0, 1]
    denoised_image = (denoised_image * 255).byte()  # Convert to [0, 255] and uint8 type

    # Save the denoised image
    denoised_image_name = os.path.basename(img_path)
    denoised_image_path = os.path.join(denoised_image_dir, denoised_image_name)
    write_jpeg(denoised_image, denoised_image_path)

    print(f"Denoised image saved to {denoised_image_path}")
