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


# Visualization function
def show(tensor, name, ch=1, size=(28, 28), num=16):
    ''' To Convert (28,28) to (128, 1, 28, 28) using (-1,ch,*size) where -1 represents 128 which is the batch size and *size is 28x28.
        Detach the tensor and store inside CPU.
        Nrows is the number of rows from 16.
        Permute is the reshape function.
    '''
    # resized_tensor = tensor#F.interpolate(tensor, size=(28, 28), mode='bilinear', align_corners=False)
    # # print("Shape after resizing:", resized_tensor.shape)  # Check the shape after resizing
    # data = resized_tensor.detach().cpu().view(-1, ch, *size)  # 128 x 1 x 28 x 28
    # grid = make_grid(data[:num], nrow=4).permute(1, 2, 0)  # 1 x 28 x 28  = 28 x 28 x 1
    # plt.imshow(grid, cmap="gray")
    # plt.savefig(name)

    single_image = tensor[0].detach().cpu().view(ch, *size)  # Choose index 0 for the first image
    plt.imshow(single_image.squeeze().numpy(), cmap="gray")
    plt.axis('off')
    plt.savefig(name)
    plt.show()


def load_noise_as_tensor(image_path):
    # Define your image transformation
    transform = transforms.Compose([
        transforms.Resize((128, 64)),  # Resize image to match model input size
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        #transforms.Normalize((0.5,), (0.5,))  # Normalize to range [-1, 1] if using Tanh in output
    ])
    image = Image.open(image_path).convert('L')  # Convert to grayscale if required by the model
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor

def load_image_as_tensor(image_path):
    # Define your image transformation
    transform = transforms.Compose([
        transforms.Resize((1024, 784)),  # Resize image to match model input size
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        # transforms.Normalize((0.5,), (0.5,))  # Normalize to range [-1, 1] if using Tanh in output
    ])
    image = Image.open(image_path).convert('L')  # Convert to grayscale if required by the model
    image_tensor = transform(image)
    image_tensor = torch.squeeze(image_tensor)
    return image_tensor


# Generator
def genBlock(inp, out):
    return nn.Sequential(
        nn.Linear(inp, out),
        nn.BatchNorm1d(out),
        nn.ReLU(inplace=True)  # For complex information capturing
    )


class Generator(nn.Module):
    def __init__(self, z_dim=64, i_dim=784, h_dim=128):  # h - hidden, i=out dimension and z = input dimension
        super().__init__()
        self.gen = nn.Sequential(
            genBlock(z_dim, h_dim),  # 784, 128
            genBlock(h_dim, h_dim * 2),  # 128, 256
            genBlock(h_dim * 2, h_dim * 4),  # 256 x 512
            genBlock(h_dim * 4, h_dim * 8),  # 512, 1024
            nn.Linear(h_dim * 8, i_dim),  # 1024, 784 (28x28)
            nn.Sigmoid(),
        )

    def forward(self, noise):
        return self.gen(noise)


# Discriminator
def discBlock(inp, out):
    return nn.Sequential(
        nn.Linear(inp, out),
        nn.LeakyReLU(0.2)  # LeakyRELU prevents neurons from dying
    )


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
    gen = Generator(z_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr) # Calculating gradient
    disc = Discriminator().to(device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

    image_path = 'data/real/1.jpg'  # Adjust this to the correct path
    noise_path = 'data/noise/1.png'  # Adjust this to the correct path
    image_tensor = load_image_as_tensor(image_path)
    noise_tensor = load_noise_as_tensor(noise_path)
    print("Image Tensor Shape:", image_tensor.shape, "Noise Tensor Shape:", noise_tensor.shape)
    noise = torch.squeeze(noise_tensor)
    real = image_tensor
    # fake = gen(noise)
    # print(fake.shape)
    # show(fake, name='fake.jpg')
    # # show(image_tensor)

    for epoch in range(epochs):

        ### discriminator
        disc_opt.zero_grad() #Set gradient to zero

        cur_bs=len(real) # real: 128 x 1 x 28 x 28
        real = real.view(cur_bs, -1) # 128 x 784
        real = real.to(device)

        disc_loss = calc_disc_loss(loss_func,gen,disc,cur_bs,real,z_dim, noise)
        disc_loss.backward(retain_graph=True) #Backpropogation to calculate gradient
        disc_opt.step() #Set

        ### generator
        gen_opt.zero_grad()
        gen_loss = calc_gen_loss(loss_func,gen,disc,cur_bs,z_dim, noise) #loss
        gen_loss.backward(retain_graph=True) #backpropogate
        gen_opt.step()

        ### visualization & stats
        mean_disc_loss+=disc_loss.item()/info_step
        mean_gen_loss+=gen_loss.item()/info_step


        # if cur_step % info_step == 0 and cur_step>0:
        fake = gen(noise)
        fname = 'source/' + str(cur_step) + '_fake.jpg'
        rname = 'source/' + str(cur_step) + '_real.jpg'
        print(fname)
        show(fake, fname)
        show(real, rname)
        print(f"{epoch}: step {cur_step} / Gen loss: {mean_gen_loss} / disc_loss: {mean_disc_loss}")
        mean_gen_loss, mean_disc_loss=0,0
        cur_step+=1

