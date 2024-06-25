import os
import kornia as K
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms

def imshow(input: torch.Tensor):
    out = torchvision.utils.make_grid(input, nrow=2, padding=5)
    out_np = K.utils.tensor_to_image(out)
    plt.imshow(out_np)
    plt.axis("off")
    plt.show()

# Define a dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_files = []
        for file in sorted(os.listdir(root_dir)):
            if file.endswith(".png"):
                self.image_files.append(os.path.join(root_dir, file))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = self.image_files[index]
        img = K.io.load_image(image_path, K.io.ImageLoadType.RGB32)
        noisy_image = (img + torch.normal(torch.zeros_like(img), 0.1)).clamp(0, 1)
        return noisy_image, 


# Define the TVDenoise model
class TVDenoise(torch.nn.Module):
    def __init__(self, img):
        super().__init__()
        self.l2_term = torch.nn.MSELoss(reduction="mean")
        self.regularization_term = K.losses.TotalVariation()
        # create the variable which will be optimized to produce the noise free image
        self.clean_image = torch.nn.Parameter(data=torch.zeros_like(img).clone(), requires_grad=True)
    def forward(self, noisy_image):
        return self.l2_term(self.clean_image, noisy_image) + 0.0001 * self.regularization_term(self.clean_image)

    def get_clean_image(self):
        return self.clean_image


# Create the dataset and data loader
root_dir = '../data/training/'
transform = transforms.Compose([transforms.ToTensor()])
dataset = CustomDataset(root_dir, transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Get a sample noisy image from the data loader
sample_noisy_image = next(iter(data_loader))[0]
# Define the TVDenoise model and optimizer
tv_denoiser = TVDenoise(sample_noisy_image)

optimizer = torch.optim.Adam(tv_denoiser.parameters(), lr=0.1)

num_iters: int = 500

for i in range(num_iters):
    for noisy_images in data_loader:
        optimizer.zero_grad()
        loss = 0
        for noisy_image in noisy_images:
            loss += tv_denoiser(noisy_image).sum()
        if i % 50 == 0:
            print(f"Loss in iteration {i} of {num_iters}: {loss.item():.3f}")
        loss.backward()
        optimizer.step()


# Convert back to numpy
img_clean = K.utils.tensor_to_image(tv_denoiser.get_clean_image())

# Create the plot
fig, axs = plt.subplots(1, 2, figsize=(16, 10))
axs = axs.ravel()

axs[0].axis("off")
axs[0].set_title("Noisy image")
axs[0].imshow(K.tensor_to_image(noisy_images))

axs[1].axis("off")
axs[1].set_title("Cleaned image")
axs[1].imshow(img_clean)

plt.show()
