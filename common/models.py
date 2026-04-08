"""Defines all graph embedding models"""
from functools import reduce
import random

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

from common import utils
from common import feature_preprocess

# GNN -> concat -> MLP graph classification baseline
class BaselineMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, args):
        super(BaselineMLP, self).__init__()
        self.emb_model = SkipLastGNN(input_dim, hidden_dim, hidden_dim, args)
        self.mlp = nn.Sequential(nn.Linear(2 * hidden_dim, 256), nn.ReLU(),
            nn.Linear(256, 2))

    def forward(self, emb_motif, emb_motif_mod):
        pred = self.mlp(torch.cat((emb_motif, emb_motif_mod), dim=1))
        pred = F.log_softmax(pred, dim=1)
        return pred

    def predict(self, pred):
        return pred#.argmax(dim=1)

    def criterion(self, pred, _, label):
        return F.nll_loss(pred, label)

# Order embedder model -- contains a graph embedding model `emb_model`
class OrderEmbedder(nn.Module):
    def __init__(self, input_dim, hidden_dim, args):
        super(OrderEmbedder, self).__init__()
        self.emb_model = SkipLastGNN(input_dim, hidden_dim, hidden_dim, args)
        self.margin = args.margin
        self.use_intersection = False

        self.clf_model = nn.Sequential(nn.Linear(1, 2), nn.LogSoftmax(dim=-1))

    def forward(self, emb_as, emb_bs):
        return emb_as, emb_bs

    def predict(self, pred):
        """Predict if b is a subgraph of a (batched), where emb_as, emb_bs = pred.

        pred: list (emb_as, emb_bs) of embeddings of graph pairs

        Returns: list of bools (whether a is subgraph of b in the pair)
        """
        emb_as, emb_bs = pred

        e = torch.sum(torch.max(torch.zeros_like(emb_as,
            device=emb_as.device), emb_bs - emb_as)**2, dim=1)
        return e

    def criterion(self, pred, intersect_embs, labels):
        """Loss function for order emb.
        The e term is the amount of violation (if b is a subgraph of a).
        For positive examples, the e term is minimized (close to 0); 
        for negative examples, the e term is trained to be at least greater than self.margin.

        pred: lists of embeddings outputted by forward
        intersect_embs: not used
        labels: subgraph labels for each entry in pred
        """
        emb_as, emb_bs = pred
        e = torch.sum(torch.max(torch.zeros_like(emb_as,
            device=utils.get_device()), emb_bs - emb_as)**2, dim=1)

        margin = self.margin
        e[labels == 0] = torch.max(torch.tensor(0.0,
            device=utils.get_device()), margin - e)[labels == 0]

        relation_loss = torch.sum(e)

        return relation_loss

class SkipLastGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args):
        super(SkipLastGNN, self).__init__()
        self.batch_norm = nn.BatchNorm1d(output_dim)
        self.dropout = args.dropout
        self.n_layers = args.n_layers
        self.num_relations = args.num_relations
        self.num_bases = args.num_bases

        if len(feature_preprocess.FEATURE_AUGMENT) > 0:
            self.feat_preprocess = feature_preprocess.Preprocess(input_dim)
            input_dim = self.feat_preprocess.dim_out
        else:
            self.feat_preprocess = None

        self.pre_mp = nn.Sequential(nn.Linear(input_dim, 3*hidden_dim if
            args.conv_type == "PNA" else hidden_dim))

        conv_model = self.build_conv_model(args.conv_type, 1)
        if args.conv_type == "PNA":
            self.convs_sum = nn.ModuleList()
            self.convs_mean = nn.ModuleList()
            self.convs_max = nn.ModuleList()
        else:
            self.convs = nn.ModuleList()

        if args.skip == 'learnable':
            self.learnable_skip = nn.Parameter(torch.ones(self.n_layers,
                self.n_layers))

        for l in range(args.n_layers):
            if args.skip == 'all' or args.skip == 'learnable':
                hidden_input_dim = hidden_dim * (l + 1)
            else:
                hidden_input_dim = hidden_dim
            if args.conv_type == "PNA":
                self.convs_sum.append(conv_model(3*hidden_input_dim, hidden_dim))
                self.convs_mean.append(conv_model(3*hidden_input_dim, hidden_dim))
                self.convs_max.append(conv_model(3*hidden_input_dim, hidden_dim))
            else:
                self.convs.append(conv_model(hidden_input_dim, hidden_dim))

        post_input_dim = hidden_dim * (args.n_layers + 1)
        if args.conv_type == "PNA":
            post_input_dim *= 3
        self.post_mp = nn.Sequential(
            nn.Linear(post_input_dim, hidden_dim * 2),
            nn.Dropout(args.dropout),
            nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(args.dropout),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        self.skip = args.skip
        self.conv_type = args.conv_type

    def build_conv_model(self, model_type, n_inner_layers):
        if model_type == "GCN":
            return pyg_nn.GCNConv
        elif model_type == "GIN":
            return lambda i, h: pyg_nn.GINConv(nn.Sequential(
                nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h)))
        elif model_type == "SAGE":
            return pyg_nn.SAGEConv
        elif model_type == "graph":
            return pyg_nn.GraphConv
        elif model_type == "GAT":
            return pyg_nn.GATConv
        elif model_type == "gated":
            return lambda i, h: pyg_nn.GatedGraphConv(h, n_inner_layers)
        elif model_type == "PNA":
            return pyg_nn.SAGEConv
        elif model_type == "RGCN":
            return lambda i, h: pyg_nn.RGCNConv(i, h, self.num_relations, self.num_bases)
        else:
            print("unrecognized model type")

    def forward(self, data):
        #if data.x is None:
        #    data.x = torch.ones((data.num_nodes, 1), device=utils.get_device())

        #x = self.pre_mp(x)
        if self.feat_preprocess is not None:
            if not hasattr(data, "preprocessed"):
                data = self.feat_preprocess(data)
                data.preprocessed = True
        x, edge_index, batch = data.node_feature, data.edge_index, data.batch
        edge_type = None
        if self.conv_type == "RGCN":
            if hasattr(data, "edge_type") and data.edge_type is not None:
                edge_type = data.edge_type
            else:
                # Fallback: single relation type for all edges.
                edge_type = torch.zeros(edge_index.size(1), dtype=torch.long,
                    device=edge_index.device)
        x = self.pre_mp(x)

        all_emb = x.unsqueeze(1)
        emb = x
        for i in range(len(self.convs_sum) if self.conv_type=="PNA" else
            len(self.convs)):
            if self.skip == 'learnable':
                skip_vals = self.learnable_skip[i,
                    :i+1].unsqueeze(0).unsqueeze(-1)
                curr_emb = all_emb * torch.sigmoid(skip_vals)
                curr_emb = curr_emb.view(x.size(0), -1)
                if self.conv_type == "PNA":
                    x = torch.cat((self.convs_sum[i](curr_emb, edge_index),
                        self.convs_mean[i](curr_emb, edge_index),
                        self.convs_max[i](curr_emb, edge_index)), dim=-1)
                elif self.conv_type == "RGCN":
                    x = self.convs[i](curr_emb, edge_index, edge_type)
                else:
                    x = self.convs[i](curr_emb, edge_index)
            elif self.skip == 'all':
                if self.conv_type == "PNA":
                    x = torch.cat((self.convs_sum[i](emb, edge_index),
                        self.convs_mean[i](emb, edge_index),
                        self.convs_max[i](emb, edge_index)), dim=-1)
                elif self.conv_type == "RGCN":
                    x = self.convs[i](emb, edge_index, edge_type)
                else:
                    x = self.convs[i](emb, edge_index)
            else:
                if self.conv_type == "RGCN":
                    x = self.convs[i](x, edge_index, edge_type)
                else:
                    x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            emb = torch.cat((emb, x), 1)
            if self.skip == 'learnable':
                all_emb = torch.cat((all_emb, x.unsqueeze(1)), 1)

        emb = pyg_nn.global_add_pool(emb, batch)
        emb = self.post_mp(emb)
        emb = self.batch_norm(emb)
        out = F.log_softmax(emb, dim=1)
        return emb

    def loss(self, pred, label):
        return F.nll_loss(pred, label)
