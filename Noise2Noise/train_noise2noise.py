import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from torchmetrics.functional import structural_similarity_index_measure as ssim
from accelerate import Accelerator

# Define the actual path to your dataset directory
root_dir = '../data/training'

# Update the NoisyImageDataset class to load images from the new directory
class NoisyImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, image  # Both input and target are noisy images

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((576, 1024)),  # Resize images to match their dimensions
    transforms.ToTensor()
])

# Initialize the dataset with the actual root directory
dataset = NoisyImageDataset(root_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)  # Use smaller batch size if GPU memory is an issue

# Define the denoising CNN model
class DenoisingCNN(nn.Module):
    def __init__(self):
        super(DenoisingCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 576x1024 -> 288x512
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 288x512 -> 144x256
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 144x256 -> 72x128
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),  # 72x128 -> 144x256
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),  # 144x256 -> 288x512
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),  # 288x512 -> 576x1024
            nn.ConvTranspose2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Initialize the denoising CNN model
model = DenoisingCNN()

# Define SSIM loss function
class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, img1, img2):
        return 1 - ssim(img1, img2)

ssim_loss = SSIMLoss()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# Train the model
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
accelerator = Accelerator()
device = accelerator.device

model.to(device)

num_epochs = 60  # Increase the number of epochs

for epoch in range(num_epochs):
    running_loss = 0.0
    print(epoch, end='',flush=True)
    print("/", end='',flush=True)
    print(num_epochs, end='',flush=True)
    print(":", end='',flush=True)
 
    for i, data in enumerate(dataloader, 0):
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        print(".", end='',flush=True)
        outputs = model(inputs)
        loss = ssim_loss(outputs, targets)
#        loss.backward()
        accelerator.backward(loss)
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:    # Print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
    print("")
    scheduler.step()

print('Finished Training')

# Save the trained model
torch.save(model.state_dict(), 'denoising_cnn.pth')

