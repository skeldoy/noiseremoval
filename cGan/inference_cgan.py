import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.io import read_image, write_jpeg
from glob import glob
import os

# Define the Generator (same as the training script)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained generator model
generator = Generator().to(device)
generator.load_state_dict(torch.load('./generator_final.pth'))
generator.eval()

# Data transformation
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Input and output directories
input_dir = '../data/training'
output_dir = '../data/cgan'
os.makedirs(output_dir, exist_ok=True)

# List all images in the input directory
image_paths = glob(os.path.join(input_dir, '*.png'))

# Perform inference
with torch.no_grad():
    for img_path in image_paths:
        # Load and preprocess the image
        image = read_image(img_path).float() / 255.0
        image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

        # Generate the denoised image
        denoised_image = generator(image)

        # Post-process and save the denoised image
        denoised_image = denoised_image.squeeze(0).cpu()  # Remove batch dimension
        denoised_image = (denoised_image * 0.5 + 0.5).clamp(0, 1)  # Rescale to [0, 1]
        denoised_image = denoised_image.mul(255).byte()  # Convert to byte tensor
        output_path = os.path.join(output_dir, os.path.basename(img_path))
        write_jpeg(denoised_image, output_path)

        print(f"Denoised image saved to {output_path}")

