import torch
from tqdm import tqdm

#------------------------------------------------------------------------------


def final_loss(bce_loss, mu, logvar):
    """
    This function add the reconstruction loss (BCELoss) and the KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    
    Args:
        bce_loss: recontruction loss
        mu: the mean from the latent vector
        logvar: log variance from the latent vector
    
    Returns:
        float : The sum of the reconstruction loss (BCELoss) and the KL-Divergence

    Code from https://discuss.pytorch.org/t/variational-autoencoder-with-custom-latent-vector-dimensions-target-and-input-size-mismatch/144536
    """

    BCE = bce_loss 
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

#------------------------------------------------------------------------------

def fit(model, dataloader, optimizer, loss_function, device):
    """
    Training function

    Args:
        model : The model of the neural network
        dataloader : The dataloader with the training data
        optimizer : An optimizer
        loss_function : A loss fuction
        device : The device on which we find the model

    Returns:
        float : The value of the objective function after the current training epoch
   
    Code adapted from https://discuss.pytorch.org/t/variational-autoencoder-with-custom-latent-vector-dimensions-target-and-input-size-mismatch/144536
    """
    
    model.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(dataloader.dataset.data)/dataloader.batch_size)):
        data, _ = data
        data = data.to(device)
        data = data.view(data.size(0), -1)
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(data)
        bce_loss = loss_function(reconstruction, data)
        loss = final_loss(bce_loss, mu, logvar)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss = running_loss/len(dataloader.dataset)

    return train_loss

#------------------------------------------------------------------------------

def validate(model, dataloader, loss_function, device):
    """
    Validation function

    Args:
        model : The model of the neural network
        dataloader : The dataloader with the training data
        optimizer : An optimizer
        loss_function : A loss fuction
        device : The device on which we find the model

    Returns:
        float : The value of the objective function after the current training epoch

    Code adapted from https://discuss.pytorch.org/t/variational-autoencoder-with-custom-latent-vector-dimensions-target-and-input-size-mismatch/144536
    """

    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(dataloader.dataset.data)/dataloader.batch_size)):
            data, _ = data
            data = data.to(device)
            data = data.view(data.size(0), -1)
            reconstruction, mu, logvar = model(data)
            bce_loss = loss_function(reconstruction, data)
            loss = final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()
        
    val_loss = running_loss/len(dataloader.dataset)

    return val_loss