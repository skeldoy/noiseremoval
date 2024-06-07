import os
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

# Custom dataset
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_paths = glob(os.path.join(img_dir, '*.png'))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# Define transform without resizing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load custom dataset
inferenceset = CustomImageDataset(img_dir='../data/training/', transform=transform)
inferenceloader = DataLoader(inferenceset, batch_size=1, shuffle=True, num_workers=2)

# Define autoencoder (same as in training script)
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.enc1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.dec1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec3 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = torch.tanh(self.dec3(x))  # Using tanh instead of sigmoid
        return x

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Autoencoder().to(device)
model_path = 'autoencoder.pth'
net.load_state_dict(torch.load(model_path))
net.eval()
print(f'Model loaded from {model_path}')

# Inference and save images
def save_image(tensor, filename):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = image * 0.5 + 0.5  # Denormalize
    image = transforms.ToPILImage()(image)
    image.save(filename)

output_dir = '../data/inference/'
os.makedirs(output_dir, exist_ok=True)

dataiter = iter(inferenceloader)

for i in range(20):
    images = next(dataiter)
    images = images.to(device)
    outputs = net(images)

    outputs = outputs.cpu().detach()
    for j, output in enumerate(outputs):
        save_image(output, os.path.join(output_dir, f'image_{i * len(outputs) + j}.png'))
