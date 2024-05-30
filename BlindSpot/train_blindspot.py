import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from glob import glob
from torch.optim import Adam
import torch.nn.functional as F
import os
import random

class NoisyImageDataset(Dataset):
    def __init__(self, image_dir, transform=None, patch_size=32):
        self.image_paths = glob(os.path.join(image_dir, '*.png'))
        self.transform = transform
        self.patch_size = patch_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = read_image(img_path).float() / 255.0  # Normalize to [0, 1]

        # Apply transform if provided
        if self.transform:
            image = self.transform(image)

        # Randomly extract a patch
        c, h, w = image.shape
        y = random.randint(0, h - self.patch_size)
        x = random.randint(0, w - self.patch_size)
        patch = image[:, y:y + self.patch_size, x:x + self.patch_size]

        # Create a mask for the center pixel
        mask = torch.ones_like(patch)
        center = self.patch_size // 2
        mask[:, center, center] = 0

        # Apply the mask to the patch
        masked_patch = patch * mask

        return masked_patch, patch[:, center, center].unsqueeze(1).unsqueeze(2)

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

# Define the training function for denoising
def train_denoising(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Zero the parameter gradients 
        optimizer.zero_grad()

        # Forward pass, compute the loss and back-propagate
        output = model(data)
        loss = nn.MSELoss()(output, target)  # Use MSE loss for denoising
        total_loss += loss.item() * data.size(0)
        loss.backward()
        optimizer.step()
    
    train_loss = total_loss / len(train_loader.dataset)
    print("Train Epoch: {} \tLoss: {:.6f}".format(epoch, train_loss))

# Save the trained model to disk
def save_model(model, path):
    torch.save(model.state_dict(), path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the BlindSpotDenoising model and move it to the GPU device
model = BlindSpotDenoising().to(device)

# Set hyperparameters
lr = 0.001
epochs = 60

# Initialize the optimizer
optimizer = Adam(model.parameters(), lr=lr)

# Define a transformation for the dataset
transform = transforms.Compose([
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the training dataset
train_dataset = NoisyImageDataset(image_dir='../data/training', transform=transform)

# Create data loader for the training dataset
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Train and save the model after each epoch
for epoch in range(1, epochs + 1):
    train_denoising(model, device, train_loader, optimizer, epoch)
    save_model(model, f'denoising_model_epoch{epoch}.pth')

# Inference function to denoise an entire image
def denoise_image(model, device, image):
    model.eval()
    with torch.no_grad():
        image = image.to(device).unsqueeze(0)  # Add batch dimension
        output = model(image)
        return output.squeeze(0)  # Remove batch dimension

# Example usage for denoising a single image
import matplotlib.pyplot as plt

def load_image(image_path):
    return read_image(image_path).float() / 255.0

def show_image(image, title='Image'):
    image = image.permute(1, 2, 0).cpu().numpy()  # Convert to HWC format for plotting
    plt.imshow(image)
    plt.title(title)
    plt.show()

# Load a noisy image
noisy_image_path = '../data/noisy_example.png'
noisy_image = load_image(noisy_image_path)

# Denoise the image using the trained model
denoised_image = denoise_image(model, device, noisy_image)

# Show the original and denoised images
show_image(noisy_image, title='Noisy Image')
show_image(denoised_image, title='Denoised Image')
