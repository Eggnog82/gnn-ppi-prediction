from torch_geometric.data import Data
import os
import torch

attr_path = './Human_features/processed_temp'
graph_path = './Human_features/processed_old'
target = './Human_features/processed'

invalid_graphs = 0
total_graphs = 0

for filename in os.listdir(attr_path):

    attr_file = os.path.join(attr_path, filename)
    graph_file = os.path.join(graph_path, filename)
    target_file = os.path.join(target, filename)


    if os.path.isfile(attr_file) and filename != 'pre_filter.pt' and filename != 'pre_transform.pt':
        total_graphs += 1

        graph = torch.load(graph_file)
        attr_graph = torch.load(attr_file)
        
        # Interatomic forces scale inversely with the square of the distance
        graph.edge_attr = torch.squeeze(1/torch.square(attr_graph.edge_attr)).unsqueeze(dim=1)

        if not graph.edge_attr.shape[0] == graph.num_edges:
            invalid_graphs += 1
        else:
            torch.save(graph, target_file)

print(f"{invalid_graphs} of the {total_graphs} graphs analyzed were removed due to dimensional inconsistencies")