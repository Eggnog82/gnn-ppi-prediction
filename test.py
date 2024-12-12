import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metrics import *
from data_prepare import testloader
from models import GCNN, AttGNN, MultiHopAttGNN, WeightedAttGNN


# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.cuda("cpu")
device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")

import argparse
MODELS = ["GCNN", "AttGNN", "MultiHopAttGNN", "WeightedAttGNN"]
parser = argparse.ArgumentParser()
parser.add_argument("model", help=f"Name of the model to train from {MODELS}")
args = parser.parse_args()
assert args.model in MODELS, f"Model must be one of the following: {MODELS}"
print(f"Testing model: {args.model}")

model = None

if args.model == "GCNN":
  model = GCNN()
elif args.model == "AttGNN":
  model = AttGNN()
elif args.model == "MultiHopAttGNN":
  model = MultiHopAttGNN()
elif args.model == "WeightedAttGNN":
  model = WeightedAttGNN()

model.load_state_dict(torch.load(f"./Human_features/{args.model}.pth")) #path to load the model
model.to(device)
model.eval()
predictions = torch.Tensor()
labels = torch.Tensor()
with torch.no_grad():
    for prot_1, prot_2, label in testloader:
      prot_1 = prot_1.to(device)
      prot_2 = prot_2.to(device)
      #print("H")
      #print(torch.Tensor.size(prot_1.x), torch.Tensor.size(prot_2.x))
      output = model(prot_1, prot_2)
      predictions = torch.cat((predictions, output.cpu()), 0)
      labels = torch.cat((labels, label.view(-1,1).cpu()), 0)
labels = labels.numpy().flatten()
predictions = predictions.numpy().flatten()


loss = get_mse(labels, predictions)
acc = get_accuracy(labels, predictions, 0.5)
prec = precision(labels, predictions, 0.5)
sensitivity = sensitivity(labels, predictions,  0.5)
specificity = specificity(labels, predictions, 0.5)
f1 = f_score(labels, predictions, 0.5)
mcc = mcc(labels, predictions,  0.5)
auroc = auroc(labels, predictions)
auprc = auprc(labels, predictions)


print(f'loss : {loss}')
print(f'Accuracy : {acc}')
print(f'precision: {prec}')
print(f'Sensititvity :{sensitivity}')
print(f'specificity : {specificity}')
print(f'f-score : {f1}')
print(f'MCC : {mcc}')
print(f'AUROC: {auroc}')
print(f'AUPRC: {auprc}')