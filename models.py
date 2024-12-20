# Building model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_max_pool as gmp, global_add_pool as gap,global_mean_pool as gep,global_sort_pool
from torch_geometric.utils import dropout_adj
from torch.optim.lr_scheduler import MultiStepLR



class GCNN(nn.Module):
    def __init__(self, n_output=1, num_features_pro= 1024, output_dim=128, dropout=0.2):
        super(GCNN, self).__init__()

        print('GCNN Loaded')

        # for protein 1
        self.n_output = n_output
        self.pro1_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro1_fc1 = nn.Linear(num_features_pro, output_dim)

        # for protein 2
        self.pro2_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro2_fc1 = nn.Linear(num_features_pro, output_dim)

        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 256)
        self.fc2 = nn.Linear(256 ,64)
        self.out = nn.Linear(64, self.n_output)

    def forward(self, pro1_data, pro2_data):

        #get graph input for protein 1 
        pro1_x, pro1_edge_index, pro1_batch = pro1_data.x, pro1_data.edge_index, pro1_data.batch
        # get graph input for protein 2
        pro2_x, pro2_edge_index, pro2_batch = pro2_data.x, pro2_data.edge_index, pro2_data.batch


        x = self.pro1_conv1(pro1_x, pro1_edge_index)
        x = self.relu(x)
        
	# global pooling
        x = gep(x, pro1_batch)   

        # flatten
        x = self.relu(self.pro1_fc1(x))
        x = self.dropout(x)



        xt = self.pro2_conv1(pro2_x, pro2_edge_index)
        xt = self.relu(xt)

	# global pooling
        xt = gep(xt, pro2_batch)  

        # flatten
        xt = self.relu(self.pro2_fc1(xt))
        xt = self.dropout(xt)


	# Concatenation  
        xc = torch.cat((x, xt), 1)

        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        out = self.sigmoid(out)
        return out
        

# net = GCNN()
# print(net)

"""# GAT"""

class AttGNN(nn.Module):
    def __init__(self, n_output=1, num_features_pro= 1024, output_dim=128, dropout=0.2, heads = 1 ):
        super(AttGNN, self).__init__()

        print('AttGNN Loaded')

        self.hidden = 8
        self.heads = 1
        
        # for protein 1
        self.pro1_conv1 = GATConv(num_features_pro, self.hidden* 16, heads=self.heads, dropout=0.2)
        self.pro1_fc1 = nn.Linear(128, output_dim)


        # for protein 2
        self.pro2_conv1 = GATConv(num_features_pro, self.hidden*16, heads=self.heads, dropout=0.2)
        self.pro2_fc1 = nn.Linear(128, output_dim)

        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.out = nn.Linear(64, n_output)
        


    def forward(self, pro1_data, pro2_data):

        # get graph input for protein 1 
        pro1_x, pro1_edge_index, pro1_batch = pro1_data.x, pro1_data.edge_index, pro1_data.batch
        # get graph input for protein 2
        pro2_x, pro2_edge_index, pro2_batch = pro2_data.x, pro2_data.edge_index, pro2_data.batch
         
        
        x = self.pro1_conv1(pro1_x, pro1_edge_index)
        x = self.relu(x)
        
	# global pooling
        x = gep(x, pro1_batch)  
       
        # flatten
        x = self.relu(self.pro1_fc1(x))
        x = self.dropout(x)



        xt = self.pro2_conv1(pro2_x, pro2_edge_index)
        xt = self.relu(self.pro2_fc1(xt))
	
	# global pooling
        xt = gep(xt, pro2_batch)  

        # flatten
        xt = self.relu(xt)
        xt = self.dropout(xt)

	
	# Concatenation
        xc = torch.cat((x, xt), 1)

        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        out = self.sigmoid(out)
        return out

# net_GAT = AttGNN()
# print(net_GAT)

"""# Multi-hop GAT"""

class MultiHopAttGNN(nn.Module):
    def __init__(self, num_hops=2, n_output=1, num_features_pro= 1024, output_dim=128, dropout=0.2, heads = 1 ):
        super(MultiHopAttGNN, self).__init__()

        print('Multi-hop AttGNN Loaded')

        self.hidden = 8
        self.heads = 1
        self.num_hops = num_hops
        
        # for protein 1
        self.pro1_conv1 = GATConv(num_features_pro, self.hidden* 16, heads=self.heads, dropout=0.2)
        self.pro1_conv2 = GATConv(self.hidden * 16 * heads, self.hidden * 16, heads=self.heads, dropout=dropout)  # 2-hop
        self.pro1_conv3 = GATConv(self.hidden * 16 * heads, self.hidden * 16, heads=self.heads, dropout=dropout)  # 3-hop
        self.pro1_fc1 = nn.Linear(128, output_dim)

        # for protein 2
        self.pro2_conv1 = GATConv(num_features_pro, self.hidden*16, heads=self.heads, dropout=0.2)
        self.pro2_conv2 = GATConv(self.hidden * 16 * heads, self.hidden * 16, heads=self.heads, dropout=dropout)  # 2-hop
        self.pro2_conv3 = GATConv(self.hidden * 16 * heads, self.hidden * 16, heads=self.heads, dropout=dropout)  # 3-hop
        self.pro2_fc1 = nn.Linear(128, output_dim)

        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.out = nn.Linear(64, n_output)
        


    def forward(self, pro1_data, pro2_data):

        # get graph input for protein 1 
        pro1_x, pro1_edge_index, pro1_two_hop_edge_index, pro1_three_hop_edge_index, pro1_batch = pro1_data.x, pro1_data.edge_index, pro1_data.two_hop_edge_index, pro1_data.three_hop_edge_index, pro1_data.batch

        # get graph input for protein 2
        pro2_x, pro2_edge_index, pro2_two_hop_edge_index, pro2_three_hop_edge_index, pro2_batch = pro2_data.x, pro2_data.edge_index, pro2_data.two_hop_edge_index, pro2_data.three_hop_edge_index, pro2_data.batch

        
        x = self.pro1_conv1(pro1_x, pro1_edge_index)
        x = self.relu(x)
        x_multi = self.pro1_conv1(pro1_x, pro1_two_hop_edge_index)
        x_multi = self.relu(x_multi)
        x = x + x_multi
        if self.num_hops >= 2:
                x_multi = self.pro1_conv1(pro1_x, pro1_three_hop_edge_index)
                x_multi = self.relu(x_multi)
                x = x + x_multi

        
	# global pooling
        x = gep(x, pro1_batch)  
       
        # flatten
        x = self.relu(self.pro1_fc1(x))
        x = self.dropout(x)


        xt = self.pro2_conv1(pro2_x, pro2_edge_index)
        xt = self.relu(xt)
        xt_multi = self.pro2_conv1(pro2_x, pro2_two_hop_edge_index)
        xt_multi = self.relu(xt_multi)
        xt = xt + xt_multi
        if self.num_hops >= 2:
                xt_multi = self.pro2_conv1(pro2_x, pro2_three_hop_edge_index)
                xt_multi = self.relu(xt_multi)
                xt = xt + xt_multi

	
	# global pooling
        xt = gep(xt, pro2_batch)  

        # flatten
        # xt = self.relu(xt)
        xt = self.relu(self.pro2_fc1(xt))
        xt = self.dropout(xt)

	
	# Concatenation
        xc = torch.cat((x, xt), 1)

        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        out = self.sigmoid(out)
        return out

# net_multihop_GAT = MultiHopAttGNN()
# print(net_multihop_GAT)

class WeightedAttGNN(nn.Module):
    def __init__(self, n_output=1, num_node_features=1024, num_edge_features=1, output_dim=128, dropout=0.2):
        super(WeightedAttGNN, self).__init__()

        print('Weighted Attention GNN Loaded')

        self.hidden1 = 256
        self.hidden2 = 128
        self.heads = 2
        
        # for protein 1
        self.pro1_conv1 = TransformerConv(in_channels=num_node_features, out_channels=self.hidden1, heads=self.heads, dropout=0.1, edge_dim=num_edge_features)
        self.pro1_fc1 = nn.Linear(self.hidden1*self.heads, num_node_features)
        self.pro1_fc2 = nn.Linear(num_node_features, output_dim)


        # for protein 2
        self.pro2_conv1 = TransformerConv(in_channels=num_node_features, out_channels=self.hidden1, heads=self.heads, dropout=0.1, edge_dim=num_edge_features)
        self.pro2_fc1 = nn.Linear(self.hidden1*self.heads, num_node_features)
        self.pro2_fc2 = nn.Linear(num_node_features, output_dim)

        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.out = nn.Linear(64, n_output)

    def forward(self, pro1_data, pro2_data):
        """Takes as input the graphs of two proteins, outputs a binary classification for whether the proteins interact or not"""

        # get graph input for protein 1 
        pro1_x, pro1_edge_index, pro1_edge_attr, pro1_batch = pro1_data.x, pro1_data.edge_index, pro1_data.edge_attr, pro1_data.batch

        # get graph input for protein 2
        pro2_x, pro2_edge_index, pro2_edge_attr, pro2_batch = pro2_data.x, pro2_data.edge_index, pro2_data.edge_attr, pro2_data.batch
         
        # Convolution
        x = self.sigmoid(self.pro1_conv1(x=pro1_x, edge_index=pro1_edge_index, edge_attr=pro1_edge_attr))
        x = self.relu(self.pro1_fc1(x))

        x = x + self.sigmoid(pro1_x)
        
	# global pooling
        x = gep(x, pro1_batch)  
       
        # flatten
        x = self.relu(self.pro1_fc2(x))
        x = self.dropout(x)

        # Convolution
        xt = self.sigmoid(self.pro2_conv1(x=pro2_x, edge_index=pro2_edge_index, edge_attr=pro2_edge_attr))
        xt = self.relu(self.pro2_fc1(xt))

        xt = xt + self.sigmoid(pro2_x)
	
	# global pooling
        xt = gep(xt, pro2_batch)  

        # flatten
        xt = self.relu(self.pro2_fc2(xt))
        xt = self.dropout(xt)
	
	# Concatenation
        xc = torch.cat((x, xt), 1)

        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        out = self.sigmoid(out)
        return out

# net_weighted_GAT = WeightedAttGNN()
# print(net_weighted_GAT)