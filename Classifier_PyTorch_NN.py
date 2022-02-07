
"""
This package uses a PyTorch neural network to classify the data sets.
"""

#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

#%%Data

def load_data():
    files = []
    for file in os.listdir(r"C:\Users\porta\OneDrive\Desktop\TBPS\Data"):
        if file.endswith(".csv"):
            data_file = pd.read_csv(file)
            files.append(data_file)
    return files

#    acceptance_mc = pd.read_csv("acceptance_mc.csv")
#    jpsi = pd.read_csv("jpsi.csv")
#    jpsi_mu_k_swap = pd.read_csv("jpsi_mu_k_swap.csv")
#    jpsi_mu_pi_swap = pd.read_csv("jpsi_mu_pi_swap.csv")
#    k_pi_swap = pd.read_csv("k_pi_swap.csv")
#    phimumu = pd.read_csv("phimumu.csv")
#    pKmumu_piTok_kTop = pd.read_csv("pKmumu_piTok_kTop.csv")
#    pKmumu_piTop = pd.read_csv("pKmumu_piTop.csv")
#    psi2S = pd.read_csv("psi2S.csv")
#    total_dataset = pd.read_csv("total_dataset.csv")
#    return signals, acceptance_mc, jpsi, jpsi_mu_k_swap, jpsi_mu_pi_swap, k_pi_swap, phimumu, pKmumu_piTok_kTop, pKmumu_piTop, psi2S, total_dataset
    

#%% Data preparation for Neural Network

'''
Details of the data sets can be seen here: https://mesmith75.github.io/ic-teach-kstmumu-public/samples/
In summary, the signal data set is the B signal we're looking for as simulated by the standard model.
The rest are backgrounds that will help with the classification.
I have excluded the total_dataset as it is the raw data we will use the classifier on as well as the 
acceptance_mc as it is to do with the acceptance function in the analyses.
'''


def prepare_data_binary(files):
    """
    Prepares data files into lists of all inputs and corresponding target values for a binary classification.
    That is "signal or not signal"
    """
    signals = pd.read_csv("signal.csv")
    acceptance_mc = pd.read_csv("acceptance_mc.csv")
    total_dataset = pd.read_csv("total_dataset.csv")
    prepped_x_data = []
    prepped_y_data = []
    for data in files:
        if data.equals(signals) == True:
            data = data.to_numpy()
            for i in range(0, len(data[:,0])):
                prepped_x_data.append(np.array(data[i, 1:79]))
                prepped_y_data.append(np.array([1.]))
            
        elif data.equals(acceptance_mc):
            pass
        elif data.equals(total_dataset):
            pass            
        
        else:
            data = data.to_numpy()
            for i in range(0, len(data[:, 0])):
                prepped_x_data.append(np.array(data[i, 1:79]))
                prepped_y_data.append(np.array([0.]))
        
    return prepped_x_data, prepped_y_data

def prepare_data_multi(files):
    """
    Prepares data files into lists of all inputs and corresponding target values for a multi-classificatiom.
    I.e each target is a 1D array of zeros with a one corresponding to the class, the number of classes being
    the number of files used in the training (9).
    """
    acceptance_mc = pd.read_csv("acceptance_mc.csv")
    total_dataset = pd.read_csv("total_dataset.csv")
    prepped_x_data = []
    prepped_y_data = []
    new_files = []
    for file in files:
        if file.equals(acceptance_mc):
            pass
        elif file.equals(total_dataset):
            pass
        else:
            new_files.append(file)
    for i, data in enumerate(new_files):       
        data = data.to_numpy()
        for j in range(0, len(data[:,0])):
            prepped_x_data.append(np.array(data[j, 1:79]))
            y_label = np.zeros(9)
            y_label[i] = 1
            prepped_y_data.append(y_label)
            
    return prepped_x_data, prepped_y_data

def normalise_data(data):
    """
    Normalise data via (x-mean)/stdev
    """
    means = np.array([])
    stds = np.array([])
    for i in range(0, len(data[0][0])):
        column = [x[i] for x in data[0]]
        means = np.append(means, np.mean(column))
        if np.std(column) == 0:
            stds = np.append(stds, 1)
        else:
            stds = np.append(stds, np.std(column))
    for j in range(0, len(data[0])):
        data[0][j] = (data[0][j] - means)/stds
    
    return data

class CustomDataset(Dataset):
    def __init__(self, training_data, transform = None):
        self.x = torch.Tensor(training_data[0])
        self.y = torch.Tensor(training_data[1])
        self.n_samples = len(training_data[0])
        self.transform = transform
        
    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        x = self.x[index]
        label = self.y[index]
        
        if self.transform:
            x = self.transform(x)
            
        return (x, label)    

#%% Neural Network

device = torch.device("cpu")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(78, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 9),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

#%% Load Data

all_data_files = load_data()
training_data = prepare_data_multi(all_data_files)
norm_training_data = normalise_data(training_data)

batch_size = 1000

dataset = CustomDataset(norm_training_data)

#Split data into training set and validation set in ratio of about 70%:30%
train_set, test_set = torch.utils.data.random_split(dataset, [1000000, 430536])

train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = True)

#%% Train Neural Network
 
model = NeuralNetwork()

learning_rate = 1e-3
epochs = 100

#Multi class loss function
loss_fn = nn.CrossEntropyLoss()

#Binary class loss function
#loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in test_loader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader, model, loss_fn, optimizer)
    test_loop(test_loader, model, loss_fn)
print("Done!")
    
            
    
