import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm
import pathlib
import math
import sklearn
import torch_optimizer as optim
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from metrics import *

# device = torch.device("cuda:1") if torch.cuda.is_available() else torch.cuda("cpu")
device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")


import torch.nn as nn
import networkx as nx
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from models import GCNN, AttGNN, MultiHopAttGNN, WeightedAttGNN
from torch_geometric.data import DataLoader as DataLoader_n



# Allow the user to select which model is trained (cuts down on repeated code)
import argparse

MODELS = ["GCNN", "AttGNN", "MultiHopAttGNN", "WeightedAttGNN"]

parser = argparse.ArgumentParser()

parser.add_argument("model", help=f"Name of the model to train from {MODELS}")
parser.add_argument("--num_hops", type=int, help="Number of hops for the MultiHopAttGNN model")

args = parser.parse_args()

assert args.model in MODELS, f"Model must be one of the following: {MODELS}"

print(f"Training model: {args.model}")

model = None

if args.model == "GCNN":
  model = GCNN()
elif args.model == "AttGNN":
  model = AttGNN()
elif args.model == "MultiHopAttGNN":
  if args.num_hops is None:
    raise ValueError("The 'num_hops' argument is required when the model is MultiHopAttGNN.")
  elif args.num_hops < 2 or args.num_hops > 3:
    raise ValueError("The 'num_hops' argument must be within the bounds of [2,3].")

  model = MultiHopAttGNN(num_hops = args.num_hops)
elif args.model == "WeightedAttGNN":
  model = WeightedAttGNN()

from data_prepare import dataset, trainloader, testloader
print("Datalength")
print(len(dataset))
print(len(trainloader))
print(len(testloader))

total_samples = len(dataset)
n_iterations = math.ceil(total_samples/5)

 
#utilities
def train(model, device, trainloader, optimizer, epoch):

  print(f'Training on {len(trainloader)} samples.....')
  model.train()
  loss_func = nn.MSELoss()
  predictions_tr = torch.Tensor()
  scheduler = MultiStepLR(optimizer, milestones=[1,5], gamma=0.5)
  labels_tr = torch.Tensor()
  for count,(prot_1, prot_2, label) in enumerate(trainloader):
    # print(f"prot_1 = {prot_1}, prot_2 = {prot_2}, label = {label}")
    prot_1 = prot_1.to(device)
    prot_2 = prot_2.to(device)
    optimizer.zero_grad()
    output = model(prot_1, prot_2)
    predictions_tr = torch.cat((predictions_tr, output.cpu()), 0)
    labels_tr = torch.cat((labels_tr, label.view(-1,1).cpu()), 0)
    loss = loss_func(output, label.view(-1,1).float().to(device))
    loss.backward()
    optimizer.step()
  scheduler.step()
  labels_tr = labels_tr.detach().numpy()
  predictions_tr = predictions_tr.detach().numpy()
  acc_tr = get_accuracy(labels_tr, predictions_tr , 0.5)
  print(f'Epoch {epoch-1} / 30 [==============================] - train_loss : {loss} - train_accuracy : {acc_tr}')
    
 

def predict(model, device, loader):
  model.eval()
  predictions = torch.Tensor()
  labels = torch.Tensor()
  with torch.no_grad():
    for prot_1, prot_2, label in loader:
      prot_1 = prot_1.to(device)
      prot_2 = prot_2.to(device)
      #print(torch.Tensor.size(prot_1.x), torch.Tensor.size(prot_2.x))
      output = model(prot_1, prot_2)
      predictions = torch.cat((predictions, output.cpu()), 0)
      labels = torch.cat((labels, label.view(-1,1).cpu()), 0)
  labels = labels.numpy()
  predictions = predictions.numpy()
  return labels.flatten(), predictions.flatten()
  
  

# training 

#early stopping
n_epochs_stop = 6
epochs_no_improve = 0
early_stop = False



model.to(device)

num_epochs = 50
loss_func = nn.MSELoss()
min_loss = 100
best_accuracy = 0
optimizer =  torch.optim.Adam(model.parameters(), lr= 0.001)
for epoch in range(num_epochs):
  start_time = time.time()
  train(model, device, trainloader, optimizer, epoch+1)
  G, P = predict(model, device, testloader)
  #print( f'Predictions---------------------------------------------{P}')
  #print(f'Labels----------------------------------------------------{G}')
  loss = get_mse(G,P)
  accuracy = get_accuracy(G,P, 0.5)
  print(f'Epoch {epoch}/ {num_epochs} [==============================] - val_loss : {loss} - val_accuracy : {accuracy}, took {time.time() - start_time} seconds')
  if(accuracy > best_accuracy):
    best_accuracy = accuracy
    best_acc_epoch = epoch
    torch.save(model.state_dict(), f"./Human_features/{args.model}.pth") #path to save the model
    print("Model")
  if(loss< min_loss):
    epochs_no_improve = 0
    min_loss = loss
    min_loss_epoch = epoch
  elif loss> min_loss :
    epochs_no_improve += 1
  if epoch > 5 and epochs_no_improve == n_epochs_stop:
    print('Early stopping!' )
    early_stop = True
    break

print(f'min_val_loss : {min_loss} for epoch {min_loss_epoch} ............... best_val_accuracy : {best_accuracy} for epoch {best_acc_epoch}')
print("Model saved")

