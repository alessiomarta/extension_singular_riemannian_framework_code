import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Simplified version of the network in https://www.mdpi.com/2079-9292/12/11/2431
# "Predicting Power Generation from a Combined Cycle Power Plant Using Transformer Encoders with DNN"

class Model(nn.Module):
    def __init__(self, in_features = 4, hidden_features = 32, out_features = 1):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.linear2A = nn.Linear(hidden_features, hidden_features)
        self.linear2B = nn.Linear(hidden_features, out_features)
        self.lstm = nn.LSTMCell(input_size=out_features, hidden_size=out_features)
        self.linear_lstm = nn.Linear(1, hidden_features)
        self.linear3 = nn.Linear(hidden_features, out_features)
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2A(x))
        residual = x
        x = F.relu(self.linear2B(x))
        x, _ = self.lstm(x)
        x = F.relu(self.linear_lstm(x))
        x = F.sigmoid(self.linear3(x+residual))
        return x