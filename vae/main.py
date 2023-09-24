import torch
import torchvision
import torch.optim as optim
import argparse
import matplotlib
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image



import vae_model
from training_functions import fit, validate
from simec_algorithm import simec_dec, simec_enc

"""
Part of this code is adapted from
- https://discuss.pytorch.org/t/variational-autoencoder-with-custom-latent-vector-dimensions-target-and-input-size-mismatch/144536
- https://gist.github.com/lyndond/ce29865d34e9c041a2701652260f2f32
"""

#------------------------------------------------------------------------------

#Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#------------------------------------------------------------------------------

#Learning parameters
epochs = 2
batch_size = 256
lr = 0.01

#------------------------------------------------------------------------------

#Transforms
transform = transforms.Compose([transforms.ToTensor(),transforms.Lambda(lambda t: t/255)])

#Training and validation data
train_data = datasets.MNIST(
    root='input/data',
    train=True,
    download=True,
    transform=transform
)

validation_data = datasets.MNIST(
    root='input/data',
    train=False,
    download=True,
    transform=transform
)

#Training and validation data loaders
train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True
)
validation_loader = DataLoader(
    validation_data,
    batch_size=batch_size,
    shuffle=False
)

#------------------------------------------------------------------------------

#Load model on device
network = vae_model.LinearVAE().to(device)

#Select optimizer and loss function
optimizer = optim.Adam(network.parameters(), lr=lr)
#For autoencoders we use reduction='sum' for the BCELoss(). 
loss_function = nn.BCELoss(reduction='sum')

#------------------------------------------------------------------------------

#Train the model
train_loss = []
val_loss = []
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = fit(network, train_loader, optimizer, loss_function, device)
    print(train_epoch_loss)
    val_epoch_loss = validate(network, validation_loader, loss_function, device)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {val_epoch_loss:.4f}")

#------------------------------------------------------------------------------

decoder = vae_model.Decoder(network)
encoder = vae_model.Encoder(network)


"""
inp = torch.tensor([0.5,0.5]).to(device)
g = torch.eye(784).to(device)
simec_dec(decoder, device, inp, g, 2000)
"""

x, y = zip(*train_data)
in_ = x[0].reshape(784)
in_ = in_.type(torch.FloatTensor)
inp = in_.to(device)
g = torch.eye(2).to(device)
simec_enc(encoder, device, inp, g, 10000)
