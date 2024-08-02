import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import os
from datetime import datetime

# Hyperparameters:
img_size = 128
img_hw = img_size * img_size * 3  # For color images (3 channels)

# Parameters
epochs = 20  # Number of complete passes over the dataset
cur_step = 0
info_step = 5  # Interval for printing information
mean_gen_loss = 0
mean_disc_loss = 0
z_dim = img_hw  # Latent space dimension
lr = 0.00001
loss_func = nn.BCEWithLogitsLoss()
bs = 16  # Batch size

device = 'cuda' if torch.cuda.is_available() else 'cpu'
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = f'output/output_{timestamp}/'
os.makedirs(output_dir, exist_ok=True)

# Visualization function
def show(tensor_batch, prefix, size=(img_size, img_size)):
    ''' Visualizes a batch of image tensors.
    Detach the tensor and store it inside CPU.
    '''
    batch_size = tensor_batch.size(0)  # Get the number of images in the batch
    for i in range(batch_size):
        single_image = tensor_batch[i].detach().cpu().view(3, *size).permute(1, 2, 0)
        plt.imshow(single_image.numpy() * 0.5 + 0.5)  # Unnormalize
        plt.axis('off')
        fname = str(output_dir) + f'{prefix}_{i}.jpg'
        plt.savefig(fname)  # Save each image with an index
        plt.show()

# Custom Dataset Class
class FaceDataset(Dataset):
    def __init__(self, eyes_dir, faces_dir, transform=None):
        self.eyes_dir = eyes_dir
        self.faces_dir = faces_dir
        self.transform = transform
        self.eyes_images = sorted(os.listdir(eyes_dir))
        self.faces_images = sorted(os.listdir(faces_dir))

    def __len__(self):
        return len(self.eyes_images)

    def __getitem__(self, idx):
        eyes_path = os.path.join(self.eyes_dir, self.eyes_images[idx])
        face_path = os.path.join(self.faces_dir, self.faces_images[idx])

        eyes_image = Image.open(eyes_path).convert('RGB')  # Convert to RGB
        face_image = Image.open(face_path).convert('RGB')  # Convert to RGB

        if self.transform:
            eyes_image = self.transform(eyes_image)
            face_image = self.transform(face_image)

        return eyes_image, face_image

# Transformations
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),  # Resize to match model input size
    transforms.ToTensor(),  # Convert to PyTorch tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize to range [-1, 1]
])

# Load the dataset
eyes_dir = 'data/noise/'  # Directory containing eyes images
faces_dir = 'data/real/'  # Directory containing face images
dataset = FaceDataset(eyes_dir, faces_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)

# Generator
def genBlock(inp, out):
    return nn.Sequential(
        nn.Linear(inp, out),
        nn.BatchNorm1d(out),
        # nn.LayerNorm(out),  # Use LayerNorm instead of BatchNorm1d
        nn.ReLU(inplace=True)
    )

# Discriminator
def discBlock(inp, out):
    return nn.Sequential(
        nn.Linear(inp, out),
        nn.LeakyReLU(0.2)  # LeakyRELU prevents neurons from dying
    )

class Generator(nn.Module):
    def __init__(self, z_dim=img_hw, i_dim=img_hw, h_dim=128):
        super().__init__()
        self.gen = nn.Sequential(
            genBlock(z_dim, h_dim),
            genBlock(h_dim, h_dim * 2),
            genBlock(h_dim * 2, h_dim * 4),
            genBlock(h_dim * 4, h_dim * 8),
            nn.Linear(h_dim * 8, i_dim),
            nn.Tanh()
        )

    def forward(self, noise):
        return self.gen(noise)

class Discriminator(nn.Module):
    def __init__(self, i_dim=img_hw, h_dim=256):
        super().__init__()
        self.disc = nn.Sequential(
            discBlock(i_dim, h_dim * 4),
            discBlock(h_dim * 4, h_dim * 2),
            discBlock(h_dim * 2, h_dim),
            nn.Linear(h_dim, 1)
        )

    def forward(self, image):
        return self.disc(image)

def calc_gen_loss(loss_func, gen, disc, noise):
    fake = gen(noise)
    pred = disc(fake)
    targets = torch.ones_like(pred)
    gen_loss = loss_func(pred, targets)
    return gen_loss

def calc_disc_loss(loss_func, gen, disc, real, noise):
    fake = gen(noise)
    disc_fake = disc(fake.detach())
    disc_fake_targets = torch.zeros_like(disc_fake)
    disc_fake_loss = loss_func(disc_fake, disc_fake_targets)

    disc_real = disc(real)
    disc_real_targets = torch.ones_like(disc_real)
    disc_real_loss = loss_func(disc_real, disc_real_targets)

    disc_loss = (disc_fake_loss + disc_real_loss) / 2
    return disc_loss

if __name__ == "__main__":
    gen = Generator(z_dim=img_hw).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
    disc = Discriminator().to(device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
    n1 = 0
    for epoch in range(epochs):
        for i, (noise, real) in enumerate(dataloader):
            noise = noise.view(-1, img_hw).to(device)
            real = real.view(-1, img_hw).to(device)

            ### Discriminator
            disc_opt.zero_grad()
            disc_loss = calc_disc_loss(loss_func, gen, disc, real, noise)
            disc_loss.backward()
            disc_opt.step()

            ### Generator
            gen_opt.zero_grad()
            gen_loss = calc_gen_loss(loss_func, gen, disc, noise)
            gen_loss.backward()
            gen_opt.step()

            ### Visualization & Stats
            mean_disc_loss += disc_loss.item() / info_step
            mean_gen_loss += gen_loss.item() / info_step

            if i % info_step == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(dataloader)}], Gen Loss: {mean_gen_loss:.4f}, Disc Loss: {mean_disc_loss:.4f}")
                mean_gen_loss, mean_disc_loss = 0, 0

        # Save the last batch's generated images
        fake = gen(noise).view(-1, 3, img_size, img_size).cpu()
        real = real.view(-1, 3, img_size, img_size).cpu()
        n1 = n1 + 1
        fname = f'{n1}_fake_image_'
        rname = f'{n1}_real_image_'
        show(fake, fname, size=(img_size, img_size))
        show(real, rname, size=(img_size, img_size))

        cur_step += 1