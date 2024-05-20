import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# Define the denoising CNN model architecture
class DenoisingCNN(nn.Module):
    def __init__(self):
        super(DenoisingCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Load the denoising CNN model
model = DenoisingCNN()
model.load_state_dict(torch.load('denoising_cnn.pth'))
model.eval()

# Define image transformations for inference
transform = transforms.Compose([
    transforms.Resize((576, 1024)),  # Resize images to match the input size of the model
    transforms.ToTensor()
])

# Define the directory containing the noisy images
noisy_images_dir = '../data/all'

# Define the directory to save denoised images
denoised_images_dir = "../cleaned"
os.makedirs(denoised_images_dir, exist_ok=True)

# Loop through all noisy images in the directory
for filename in os.listdir(noisy_images_dir):
    if filename.endswith('.png'):
        print("Processing ", filename, ": ",end='',flush=True)
        # Load the noisy image
        noisy_image_path = os.path.join(noisy_images_dir, filename)
        noisy_image = Image.open(noisy_image_path).convert('RGB')

        # Apply transformations
        noisy_image_tensor = transform(noisy_image).unsqueeze(0)  # Add batch dimension

        # Perform denoising
        with torch.no_grad():
            denoised_image_tensor = model(noisy_image_tensor)

        # Convert tensor back to image
        denoised_image = transforms.ToPILImage()(denoised_image_tensor.squeeze(0).cpu())

        # Save the denoised image
        denoised_image_path = os.path.join(denoised_images_dir, filename)
        try:
            denoised_image.save(denoised_image_path)
            print("OK")
        except Exception as e:
            print("Error saving image: ", e)

print("All done")
