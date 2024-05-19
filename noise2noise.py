import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
from PIL import Image
import os

# Custom dataset to load noisy images
class NoisyImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root_dir, transform=transform)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        noisy_img, _ = self.dataset[idx]
        return noisy_img, noisy_img  # Both input and target are noisy images

# Simple CNN model for denoising
class DenoisingCNN(nn.Module):
    def __init__(self):
        super(DenoisingCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Hyperparameters
lr = 0.001
batch_size = 32
epochs = 50
image_size = 64

# Image transformations
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load dataset
dataset = NoisyImageDataset(root_dir="path_to_your_dataset", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DenoisingCNN().to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(epochs):
    for i, data in enumerate(dataloader, 0):
        noisy_imgs, _ = data
        noisy_imgs = noisy_imgs.to(device)

        optimizer.zero_grad()
        outputs = model(noisy_imgs)
        loss = criterion(outputs, noisy_imgs)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Save model
torch.save(model.state_dict(), 'denoising_cnn.pth')

# Denoise and display images
model.eval()
with torch.no_grad():
    for i in range(10):
        noisy_img, _ = dataset[i]
        noisy_img = noisy_img.unsqueeze(0).to(device)
        denoised_img = model(noisy_img).squeeze().cpu().numpy().transpose(1, 2, 0)

        plt.figure(figsize=(6, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(noisy_img.squeeze().cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5)
        plt.title('Noisy Image')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(denoised_img * 0.5 + 0.5)
        plt.title('Denoised Image')
        plt.axis('off')
        plt.show()

