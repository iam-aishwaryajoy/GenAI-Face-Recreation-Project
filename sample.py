# Import the libraries
import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Parameters
epochs = 20
cur_step = 0
info_step = 5  # Current information printing
mean_gen_loss = 0
mean_disc_loss = 0
z_dim = 64 # Change this to match the input dimension
lr = 0.00001
loss_func = nn.BCEWithLogitsLoss()
bs = 128 # Batch size
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def show(tensor, name, ch=1, size=(28, 28)):
    ''' Visualizes a single image tensor.
    Detach the tensor and store it inside CPU.
    '''
    # Select the first image from the batch (or any specific image index)
    single_image = tensor[0].detach().cpu().view(ch, *size)  # Choose index 0 for the first image
    plt.imshow(single_image.squeeze().numpy(), cmap="gray")
    plt.axis('off')
    plt.savefig(name)
    plt.show()
    
def calc_gen_loss(loss_func, gen, disc, number, z_dim, noise): #number - no of elements need to process, z_dim = latent space dim
   fake = gen(noise)
   pred = disc(fake)
   targets=torch.ones_like(pred)
   gen_loss=loss_func(pred,targets)

   return gen_loss


def calc_disc_loss(loss_func, gen, disc, number, real, z_dim, noise):
   fake = gen(noise)
   disc_fake = disc(fake.detach()) #Detach is used so that we donot change parameters of generator, we only need to change parameters of discriminator.
   disc_fake_targets=torch.zeros_like(disc_fake)
   disc_fake_loss=loss_func(disc_fake, disc_fake_targets)

   disc_real = disc(real)
   disc_real_targets=torch.ones_like(disc_real)
   disc_real_loss=loss_func(disc_real, disc_real_targets)

   disc_loss=(disc_fake_loss+disc_real_loss)/2

   return disc_loss


def load_noise_as_tensor(image_path):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # Resize image to fit input dimension for generator
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize to range [-1, 1]
    ])
    image = Image.open(image_path).convert('L')  # Ensure grayscale
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    return image_tensor

def load_image_as_tensor(image_path):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # Resize image to fit output dimension of generator
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize to range [-1, 1]
    ])
    image = Image.open(image_path).convert('L')  # Ensure grayscale
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    return image_tensor

# Generator
def genBlock(inp, out):
    return nn.Sequential(
        nn.Linear(inp, out),
        #nn.BatchNorm1d(out),
        nn.LayerNorm(out),  # Use LayerNorm instead of BatchNorm1d
        nn.ReLU(inplace=True)  # For complex information capturing
    )
# Discriminator
def discBlock(inp, out):
    return nn.Sequential(
        nn.Linear(inp, out),
        nn.LeakyReLU(0.2)  # LeakyRELU prevents neurons from dying
    )

class Generator(nn.Module):
    def __init__(self, z_dim=64, i_dim=784, h_dim=128):  # Modify dimensions if necessary
        super().__init__()
        self.gen = nn.Sequential(
            genBlock(z_dim, h_dim),
            genBlock(h_dim, h_dim * 2),
            genBlock(h_dim * 2, h_dim * 4),
            genBlock(h_dim * 4, h_dim * 8),
            nn.Linear(h_dim * 8, i_dim),
            nn.Tanh()  # Use Tanh to output values between -1 and 1
        )

    def forward(self, noise):
        return self.gen(noise)

class Discriminator(nn.Module):
    def __init__(self, i_dim=784, h_dim=256):
        super().__init__()
        self.disc = nn.Sequential(
            discBlock(i_dim, h_dim * 4),  # 784, 1024
            discBlock(h_dim * 4, h_dim * 2),  # 1024, 512
            discBlock(h_dim * 2, h_dim),  # 512, 256
            nn.Linear(h_dim, 1)  # 256, 1
        )

    def forward(self, image):
        return self.disc(image)

if __name__ == "__main__":
    gen = Generator(z_dim=784).to(device)  # Adjust z_dim to match the flattened input size
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
    disc = Discriminator().to(device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

    image_path = 'data/real/1.jpg'  # Path to face image
    noise_path = 'data/noise/1.png'  # Path to eyes image

    real = load_image_as_tensor(image_path).to(device).view(-1, 784)  # Flatten image
    noise = load_noise_as_tensor(noise_path).to(device).view(-1, 784)  # Flatten image

    for epoch in range(epochs):
        ### Discriminator
        disc_opt.zero_grad()

        disc_loss = calc_disc_loss(loss_func, gen, disc, len(real), real, z_dim, noise)
        disc_loss.backward(retain_graph=True)
        disc_opt.step()

        ### Generator
        gen_opt.zero_grad()
        gen_loss = calc_gen_loss(loss_func, gen, disc, len(real), z_dim, noise)
        gen_loss.backward()
        gen_opt.step()

        ### Visualization & Stats
        mean_disc_loss += disc_loss.item() / info_step
        mean_gen_loss += gen_loss.item() / info_step

        if cur_step % info_step == 0:
            fake = gen(noise)
            fname = 'output/' + str(cur_step) + '_fake.jpg'
            rname = 'output/' + str(cur_step) + '_real.jpg'
            show(fake, fname, size=(28, 28))  # Adjust size if needed
            show(real, rname, size=(28, 28))
            print(f"{epoch}: step {cur_step} / Gen loss: {mean_gen_loss} / disc_loss: {mean_disc_loss}")
            mean_gen_loss, mean_disc_loss = 0, 0

        cur_step += 1