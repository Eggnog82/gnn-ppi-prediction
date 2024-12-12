
# note that this custom dataset is not prepared on the top of geometric Dataset(pytorch's inbuilt)
import os
import torch
import glob
import numpy as np 
import random
import math
from os import listdir
from os.path import isfile, join
from torch_sparse import coalesce
import sys


def compute_multi_hop_edges(edge_index, num_hops, num_nodes):
    adj = edge_index_to_adjacency_matrix(edge_index, num_nodes)
    multi_hop_adj = torch.matrix_power(adj, num_hops)  # Compute multi-hop adjacency
    multi_hop_edge_index = adjacency_matrix_to_edge_index(multi_hop_adj)
    return multi_hop_edge_index

def edge_index_to_adjacency_matrix(edge_index, num_nodes):
    adj = torch.zeros((num_nodes, num_nodes))
    adj[edge_index[0], edge_index[1]] = 1
    return adj

def adjacency_matrix_to_edge_index(adj):
    edge_index = torch.nonzero(adj, as_tuple=False).t()
    return edge_index


processed_dir="./Human_features/processed/"
npy_file = "./Human_features/npy_file_new(human_dataset).npy"
npy_ar = np.load(npy_file)
print(npy_ar)
print(npy_ar.shape)
from torch.utils.data import Dataset as Dataset_n
from torch_geometric.data import DataLoader as DataLoader_n

class LabelledDataset(Dataset_n):
    def __init__(self, npy_file, processed_dir):
      self.npy_ar = np.load(npy_file)
      self.processed_dir = processed_dir
      self.protein_1 = self.npy_ar[:,2]
      self.unique_protein_1 = np.unique(self.protein_1)
      print(f"npy_ar column 2 is {self.protein_1}, length = {len(self.protein_1)}, num_unique = {len(self.unique_protein_1)}", flush=True)
      self.protein_2 = self.npy_ar[:,5]
      self.unique_protein_2 = np.unique(self.protein_2)
      print(f"npy_ar column 5 is {self.protein_2}, length = {len(self.protein_2)}, num_unique = {len(self.unique_protein_2)}")
      self.label = self.npy_ar[:,6].astype(float)
      self.n_samples = self.npy_ar.shape[0]

      self.num_hops = 3
      self.multi_hop_edges = {}  # Dictionary to store precomputed multi-hop edges
      self.two_hop_edges = {}
      self.three_hop_edges = {}
      self.precompute_multi_hop_edges()


    def precompute_edges_for_protein(protein, num_hops):
        prot_path = os.path.join(self.processed_dir, protein + ".pt")
        if not glob.glob(prot_path):  # Skip missing files
            print(f"Protein file not found: {prot_path}", flush=True)
            return None

        protein_data = torch.load(glob.glob(prot_path)[0])
        edge_index, num_nodes = protein_data.edge_index, protein_data.num_nodes

        # Compute multi-hop edges for the desired hops
        edge_index = compute_multi_hop_edges(edge_index, 2, num_nodes)

        return protein, edge_index
    def precompute_multi_hop_edges(self):
      all_proteins = np.unique(np.concatenate([self.protein_1, self.protein_2]))
      print(f"Precomputing multi-hop edges for {len(all_proteins)} unique proteins...", flush=True)
      
      for protein in all_proteins:
          prot_path = os.path.join(self.processed_dir, protein + ".pt")
          if not glob.glob(prot_path):  # Skip missing files
              print(f"Protein file not found: {prot_path}")
              continue

          protein_data = torch.load(glob.glob(prot_path)[0])
          edge_index, num_nodes = protein_data.edge_index, protein_data.num_nodes
          self.two_hop_edges[protein] = compute_multi_hop_edges(edge_index, 2, num_nodes)
          self.three_hop_edges[protein] = compute_multi_hop_edges(edge_index, 3, num_nodes)

    def __len__(self):
      return(self.n_samples)

    def __getitem__(self, index):
      prot_1 = os.path.join(self.processed_dir, self.protein_1[index]+".pt")
      prot_2 = os.path.join(self.processed_dir, self.protein_2[index]+".pt")
      #print(f'Second prot is {prot_2}')
      prot_1 = torch.load(glob.glob(prot_1)[0])
      #print(f'Here lies {glob.glob(prot_2)}')
      prot_2 = torch.load(glob.glob(prot_2)[0])

      # Retrieve precomputed multi-hop edges

      prot_1_two_hop = self.two_hop_edges.get(self.protein_1[index], None)
      prot_1_three_hop = self.three_hop_edges.get(self.protein_1[index], None)
      prot_2_two_hop = self.two_hop_edges.get(self.protein_2[index], None)
      prot_2_three_hop = self.three_hop_edges.get(self.protein_2[index], None)


      # Add multi-hop edges to the Data objects
      if prot_1_two_hop is not None:
        prot_1.two_hop_edge_index = prot_1_two_hop
        prot_1.three_hop_edge_index = prot_1_three_hop
      else:
          print(f"Warning: Multi-hop edges not found for protein {self.protein_1[index]}")

      if prot_2_two_hop is not None:
        prot_2.two_hop_edge_index = prot_2_two_hop
        prot_2.three_hop_edge_index = prot_2_three_hop

      else:
          print(f"Warning: Multi-hop edges not found for protein {self.protein_2[index]}")

      return prot_1, prot_2, torch.tensor(self.label[index])



dataset = LabelledDataset(npy_file = npy_file ,processed_dir= processed_dir)

final_pairs =  np.load(npy_file)
size = final_pairs.shape[0]
print("Size is : ")
print(size)
seed = 42
torch.manual_seed(seed)
#print(math.floor(0.8 * size))
#Make iterables using dataloader class  
trainset, testset = torch.utils.data.random_split(dataset, [math.floor(0.8 * size), size - math.floor(0.8 * size) ])
#print(trainset[0])
trainloader = DataLoader_n(dataset= trainset, batch_size= 4, num_workers = 0)
testloader = DataLoader_n(dataset= testset, batch_size= 4, num_workers = 0)
# trainloader = DataLoader_n(dataset= trainset, batch_size= 8, num_workers = 0)
# testloader = DataLoader_n(dataset= testset, batch_size= 8, num_workers = 0)
print("Length")
print(len(trainloader))
print(len(testloader))


