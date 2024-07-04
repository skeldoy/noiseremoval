import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class NoisyImagesDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_filenames = os.listdir(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.image_filenames[idx])
        original_image = Image.open(image_path).convert('RGB')
        self.original_size = original_image.size
        print(self.original_size)
        if self.transform:
            image = self.transform(original_image)
        return image, image  # CAE needs both input and target to be same

class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.AdaptiveMaxPool2d((None, None)),
        )
        #self.decoder = nn.Sequential(
        #    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #    nn.Conv2d(64, 3, kernel_size=1, stride=1),
        #    nn.ReLU(True),
        #)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return torch.sigmoid(x)

transform = transforms.Compose([
    #transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

dataset = NoisyImagesDataset('../data/training/', transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

model = CAE()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for images, _ in dataloader:
        outputs = model(images)
        print(images.shape)
        print(outputs.shape)
        loss = criterion(outputs, images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
torch.save(model.state_dict(), 'cae_model.pth')

model = CAE()
model.load_state_dict(torch.load('cae_model.pth'))
model.eval()

# Create the denoised directory if it doesn't exist
os.makedirs('../data/denoised', exist_ok=True)

for image_filename in os.listdir('../data/training/'):
    image_path = os.path.join('../data/training/', image_filename)
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # add batch dimension
    denoised_image = model(image)
    denoised_image = denoised_image.squeeze().permute(1, 2, 0).detach().numpy()
    denoised_image = (denoised_image * 255).astype('uint8')
    denoised_image = Image.fromarray(denoised_image)
    denoised_image.save(os.path.join('../data/denoised/', image_filename))
