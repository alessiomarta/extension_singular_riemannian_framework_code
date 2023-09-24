import torch
import torch.nn as nn
import torch.nn.functional as F

#------------------------------------------------------------------------------

class LinearVAE(nn.Module):
    """
    Model of a simple linear VAE
    """
    def __init__(self, in_features = 784, latent_features = 2):
        super(LinearVAE, self).__init__()

        #The dimension of the latent space
        self.latent_features = latent_features
        self.in_features = in_features
 
        #Encoder
        self.enc1 = nn.Linear(in_features=self.in_features, out_features=512)
        self.enc2 = nn.Linear(in_features=512, out_features=self.latent_features*2)
 
        #Decoder 
        self.dec1 = nn.Linear(in_features=self.latent_features, out_features=512)
        self.dec2 = nn.Linear(in_features=512, out_features=self.in_features)

    def reparameterize(self, mu, log_var):
        """
        Args:
            mu: mean from the encoder's latent space
            log_var: log variance from the encoder's latent space
        """
        #Compute the standard deviation
        std = torch.exp(0.5*log_var)
        
        #Generate random numbers with randn_like, as we need the same size
        eps = torch.randn_like(std) 
        
        #Sampling as if coming from the input space
        sample = mu + (eps * std)

        return sample
 
    def forward(self, x):
        """
        The forward function of the model
        Args:
            x (torch.Tensor): The inpot of the network
        Returns:
            torch.Tensor, float,float: The tensor containing the reconstruced data and the pair (mean, log_var) from the latent space
        """
        #Encoding
        x = F.relu(self.enc1(x))
        x = self.enc2(x).view(-1, 2, self.latent_features)
        
        #Get mu and log_var: first feature values as mean and second one as variance
        mu = x[:, 0, :]
        log_var = x[:, 1, :]
        
        #Get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
 
        #Decoding
        x = F.relu(self.dec1(z))
        reconstruction = torch.sigmoid(self.dec2(x))
        return reconstruction, mu, log_var

#------------------------------------------------------------------------------

class Decoder():
    """
    Model of the decoder part of linear VAE
    """
    def __init__(self, model):
        #Decoder 
        self.dec1 = model.dec1
        self.dec2 = model.dec2

    def forward(self, x):
        #Decoding
        x = F.relu(self.dec1(x))
        reconstruction = torch.sigmoid(self.dec2(x))
        
        return reconstruction


#------------------------------------------------------------------------------

class Encoder():
    """
    Model of the encoder part of linear VAE
    """
    def __init__(self, model):
        #Encoder 
        self.enc1 = model.enc1
        self.enc2 = model.enc2
        self.latent_features = model.latent_features

    def forward(self, x):
        #Ecnoding
        x = F.relu(self.enc1(x))
        x = self.enc2(x).view(2, self.latent_features)
        x = x[:,0]

        return x

#------------------------------------------------------------------------------

def subnetwork(model, start_layer_idx, end_layer_idx):
    subnetwork = nn.Sequential()
    for idx, layer in enumerate(list(model)[start_layer_idx: end_layer_idx+1]):
        subnetwork.add_module("layer_{}".format(idx), layer)
    return subnetwork

