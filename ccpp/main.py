import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import csv
from model import Model
import matplotlib.pyplot as plt
from simec_algorithm import simec

#Select the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#--------------------------------------------------------------------------------------------

def normalize_dataset(data):

    max_values = []
    for i in range(len(data[0])):
        max_values.append(np.max(data[:,i:i+1]))

    max_values = np.array(max_values)

    min_values = []
    for i in range(len(data[0])):
        min_values.append(np.min(data[:,i:i+1]))

    min_values = np.array(min_values)

    for elem in data:
        for i in range(len(data[0])):
            elem[i] = (elem[i] - min_values[i]) / (max_values[i]-min_values[i])
    
    return data, max_values, min_values

#--------------------------------------------------------------------------------------------

def load_and_normalize_dataset(dataset_file = "combined.csv", validation_proportion = 0.3):
    rows = []

    with open(dataset_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';',quoting=csv.QUOTE_NONNUMERIC)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                col_names = row
                line_count += 1
            else:
                rows.append(row)
                line_count += 1

    data = np.asarray(rows,dtype=np.float32)
    data, max_values, min_values = normalize_dataset(data)  
    validation_idx = int(len(data)*(1.-validation_proportion))

    x_train = data[:validation_idx,0:4]
    y_train = data[:validation_idx,3:4]

    x_validate = data[validation_idx:,0:4]
    y_validate = data[validation_idx:,3:4]

    return x_train, y_train, x_validate, y_validate, max_values, min_values

#--------------------------------------------------------------------------------------------

def train_model(model,n_epochs,batch_size,x_train, y_train, x_validate, y_validate):
    loss_fn = nn.MSELoss()  # mean square error
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    batch_start = torch.arange(0, len(x_train), batch_size)

    history = []

    for epoch in tqdm(range(n_epochs)):
        model.train()
        for start in batch_start:
            # take a batch
            X_batch = x_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]
            # forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()

        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(x_validate)
        mse = loss_fn(y_pred, y_validate)
        mse = float(mse)
        history.append(mse)

#--------------------------------------------------------------------------------------------

def build_equivalence_class(model, starting_point, delta):

    g = torch.eye(1)
    g = g.to(device)
    eq_class = []

    point = starting_point
    eq_class.append(point.cpu().detach().numpy())

    for i in range(1000):
        point = simec(model, point, g, 1, delta=1e-4)
        eq_class.append(point.cpu().detach().numpy())
        print(point)
        print(model(point))

    eq_class = np.array(eq_class)

    for i in range(len(eq_class)):
        eq_class[i] = (max_values[:4]-min_values[:4])*eq_class[i]+min_values[:4]
    
    return eq_class

#--------------------------------------------------------------------------------------------

def plot_2d_projections(eq_class):

    plt.title("T-V")
    plt.scatter(eq_class[:,0], eq_class[:,1], color="red")
    plt.show()
    plt.clf()

    plt.title("T-AP")
    plt.scatter(eq_class[:,0], eq_class[:,2], color="red")
    plt.show()
    plt.clf()

    plt.title("T-RH")
    plt.scatter(eq_class[:,0], eq_class[:,3], color="red")
    plt.show()
    plt.clf()

    plt.title("V-RH")
    plt.scatter(eq_class[:,1], eq_class[:,3], color="red")
    plt.show()
    plt.clf()

#--------------------------------------------------------------------------------------------

def plot_3d_projection(eq_class):

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(eq_class[:,0], eq_class[:,1], eq_class[:,2])
    plt.show()
    plt.clf()



#Load training and validation data
x_train,y_train,x_validate,y_validate,max_values,min_values = load_and_normalize_dataset()
X_train = torch.from_numpy(x_train).to(device)
Y_train = torch.from_numpy(y_train).to(device)
X_test = torch.from_numpy(x_validate).to(device)
Y_test= torch.from_numpy(y_validate).to(device)

#Build and train the model
model = Model()
model = model.to(device)
train_model(model, 500, 100, X_train, Y_train, X_test, Y_test)

#Build equivalence class
starting_point = X_train[0]
eq_class = build_equivalence_class(model, starting_point, 1e-4)
plot_2d_projections(eq_class)
plot_3d_projection(eq_class)









