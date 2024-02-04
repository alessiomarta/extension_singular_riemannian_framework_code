import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from algorithms import simec
from model import Network
import matplotlib.pyplot as plt
import numpy as np

#Select the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Training settings
n_epochs = 10
batch_size_train = 128
batch_size_test = 64
	
random_seed = 1
#torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


#--------------------------------------------------------------------------------------------

def load_dataset():

    #The coordinates of the features space are normalized with min=-.42 and max=2.8
    train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('files/', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
    batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('files/', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
    batch_size=batch_size_test, shuffle=True)

    return train_loader, test_loader

#--------------------------------------------------------------------------------------------

def train_epoch(network, train_loader, train_losses, train_counter, epoch, learning_rate, momentum, log_interval):

  optimizer = optim.SGD(network.parameters(), lr=learning_rate,momentum=momentum)
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = network(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))

#--------------------------------------------------------------------------------------------

def test(network, test_loader, test_losses):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

#--------------------------------------------------------------------------------------------

def train(network,train_loader, train_losses, train_counter, test_losses, n_epochs, learning_rate = 0.01, momentum = 0.5, log_interval = 10, save_network = False):

    for epoch in range(1, n_epochs + 1):
        train_epoch(network,train_loader,train_losses,train_counter, epoch, learning_rate, momentum, log_interval)
        test(network, test_loader, test_losses)

    if save_network:
        torch.save(network.state_dict(), 'model.pth')

#--------------------------------------------------------------------------------------------

def save_img_to_file(network,pictures_folder,imgs_list):
    for i,img in enumerate(imgs_list):
        print("Prediction")
        #Print the probabilities for the image to be classified as one of the ten digits 
        print(np.exp(network(img).detach().numpy()))

        xp = img.detach().numpy()
        fig = plt.figure()
        plt.imshow(xp[0], cmap='gray', interpolation='none')
        filename = pictures_folder+"/"+str(i)
        plt.savefig(filename)
        plt.close()

#--------------------------------------------------------------------------------------------

#Load data
train_loader, test_loader = load_dataset()
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

#Create and train network
network = Network()
n_epochs = 1
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
train(network, train_loader, train_losses, train_counter, test_losses, n_epochs, learning_rate = 0.01, momentum = 0.5, log_interval = 10)

#Run Simexp
g = torch.eye(10)
x1 = example_data[2:3][0]
x1.requires_grad = True
imgs_list = simec(network,  x1, g, 1000, save_every_n_steps = 100, delta = 1e-1)
save_img_to_file(network,"results",imgs_list)
