import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

# Custom dataset to add noise to images
class NoisyImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root_dir, transform=transform)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        clean_img, _ = self.dataset[idx]
        noisy_img = self.add_noise(clean_img)
        return noisy_img, clean_img

    def add_noise(self, img):
        noise_factor = 0.5
        img = np.array(img)
        noisy_img = img + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=img.shape)
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)

# Generator model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

# Hyperparameters
lr = 0.0002
batch_size = 64
epochs = 100
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

# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = Generator().to(device)
D = Discriminator().to(device)

# Loss function and optimizers
criterion = nn.BCELoss()
optimizerD = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

# Labels
real_label = 1.
fake_label = 0.

# Training loop
for epoch in range(epochs):
    for i, data in enumerate(dataloader, 0):
        # Update Discriminator
        D.zero_grad()
        real_cpu = data[1].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        output = D(real_cpu).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        noise = data[0].to(device)
        fake = G(noise)
        label.fill_(fake_label)
        output = D(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        # Update Generator
        G.zero_grad()
        label.fill_(real_label)
        output = D(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

    print(f'[{epoch+1}/{epochs}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')

# Save models
torch.save(G.state_dict(), 'generator.pth')
torch.save(D.state_dict(), 'discriminator.pth')

# Denoise and display images
G.eval()
with torch.no_grad():
    for i in range(10):
        noisy_image, clean_image = dataset[i]
        noisy_image = noisy_image.unsqueeze(0).to(device)
        denoised_image = G(noisy_image).squeeze().cpu().numpy().transpose(1, 2, 0)
        clean_image = clean_image.numpy().transpose(1, 2, 0)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(noisy_image.squeeze().cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5)
        plt.title('Noisy Image')
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(clean_image * 0.5 + 0.5)
        plt.title('Clean Image')
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(denoised_image * 0.5 + 0.5)
        plt.title('Denoised Image')
        plt.axis('off')
        plt.show()

