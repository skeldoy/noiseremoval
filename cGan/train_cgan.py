import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.io import read_image, write_jpeg
from PIL import Image
from torchvision.utils import save_image
from glob import glob
import os
from torch.utils.data import DataLoader


# Define the Generator
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, 64 * 8, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            # state size: (ngf*8) x 5 x 5
            nn.ConvTranspose2d(64 * 8, 64 * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            # state size: (ngf*4) x 10 x 10
            nn.ConvTranspose2d(64 * 4, 64 * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            # state size: (ngf*2) x 20 x 20
            nn.ConvTranspose2d(64 * 2, 3, kernel_size=5, stride=2, padding=1),
            nn.Tanh()
            # state size: (nc) x 100 x 400
        )

    def forward(self, x):
        return self.main(x)

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (3) x 1024 x 576
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (64) x 512 x 288
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (128) x 256 x 144
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (256) x 128 x 72
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (512) x 64 x 36
            nn.Conv2d(512, 1024, kernel_size=4, stride=1, padding=2),
            #nn.BatchNorm2d(1024),
            #nn.LeakyReLU(0.2, inplace=True),
            # state size: (1024) x 32 x 18
            nn.Conv2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (512) x 16 x 9
            nn.Flatten(),
            nn.Linear(1024 * 16 * 18, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.main(x)

# Data loader
transform = transforms.Compose([
    transforms.Resize((1024, 576)), # Resize images to 1024x576
    transforms.ToTensor(), # Convert images to tensors
])


# Function to load images directly from a directory
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform=transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = read_image(img_path).float() / 255.0
        if self.transform:
            image = self.transform(image)
        return image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
latent_dim = 100 # Define the dimension of noise vector
image_height = 1024 # Define the height of your input images
image_width = 576 # Define the width of your input images
num_epochs = 50
batch_size = 32
lr = 0.0002
beta1 = 0.5

# Data loading and transformation
# Data loading and transformation for 1024x576 images
transform = transforms.Compose([
    transforms.Resize((image_height, image_width)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

image_paths = glob('../data/training/*.png')
train_dataset = CustomDataset(image_paths, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# Initialize Generator and Discriminator
generator = Generator(latent_dim=latent_dim).to(device)
discriminator = Discriminator().to(device)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# Loss function
criterion = nn.BCELoss()

# Training loop
for epoch in range(num_epochs):
    for i, images in enumerate(train_loader):
        images = images.to(device)

        # Create batch size
        b_size = images.size(0)

        # Adversarial ground truths
        real_labels = torch.ones((b_size, 1)).to(device)
        fake_labels = torch.zeros((b_size, 1)).to(device)

        # Train Generator
        optimizer_G.zero_grad()

        z = torch.randn((b_size, latent_dim, 1, 1), device=device)

        fake_images = generator(z)
        outputs = discriminator(fake_images)

        g_loss = criterion(outputs, real_labels)

        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()

        real_loss = criterion(discriminator(images), real_labels)
        fake_loss = criterion(discriminator(fake_images.detach()), fake_labels)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()


        # Train Generator
        z = torch.randn(images.size(0), 3, image_size, image_size).to(device)
        fake_images = generator(z)
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        if (i+1) % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Step [{i+1}/{len(train_loader)}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}, D(x): {real_score.mean().item():.4f}, D(G(z)): {fake_score.mean().item():.4f}")

    if (epoch+1) % 10 == 0:
        save_image(fake_images.data, f'./denoised_images/fake_images_{epoch+1}.png')

# Save the final model
torch.save(generator.state_dict(), './generator_final.pth')
torch.save(discriminator.state_dict(), './discriminator_final.pth')

