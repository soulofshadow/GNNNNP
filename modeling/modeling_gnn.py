from torch_geometric.nn import GATConv, Sequential, GATv2Conv
from torch_geometric.nn.dense import Linear

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.layers import GELU, GraphNorm
activation_classes = {'gelu': GELU, 'relu': nn.ReLU, 'tanh': nn.Tanh}

import math

class GATLayer(nn.Module):
    def __init__(self, dim, num_heads, dropout, norm, graph_size, add_edge_attr):
        super(GATLayer, self).__init__()

        self.norm = norm
        if self.norm:
            # GraphNorm(dim, graph_size)
            self.ln = nn.LayerNorm(dim)
        
        if add_edge_attr == 'semantic':
            self.gat = GATv2Conv(dim, dim//num_heads, edge_dim=dim, heads=num_heads, dropout=dropout)
        elif add_edge_attr == 'solo_label':
            self.gat = GATv2Conv(dim, dim//num_heads, edge_dim=1, heads=num_heads, dropout=dropout)
        else:
            self.gat = GATv2Conv(dim, dim//num_heads, heads=num_heads, dropout=dropout)

    def forward(self, x, edge_index, edge_attr):
        if self.norm:
            x = self.ln(x)
        x = self.gat(x, edge_index, edge_attr)
        return x

class GATNet(nn.Module):
    def __init__(self, dim, gnn_layers, gnn_heads, graph_size, gnn_norm=False, gnn_residual='no', add_edge_attr='no', dropout_gnn=0.2, activation='gelu'):
        super(GATNet, self).__init__()

        self.gnn_layers = gnn_layers

        self.dim = dim

        self.gnn_heads = gnn_heads
        self.gnn_norm = gnn_norm
        self.graph_size = graph_size
        self.gnn_dropout = dropout_gnn
        self.gnn_residual = gnn_residual
        self.add_edge_attr = add_edge_attr

        if self.gnn_layers != 1:
            self.activation = activation_classes[activation.lower()]()

        if self.gnn_residual == 'linear':
            self.lin = nn.Linear(self.dim, self.dim)
        
        self.layers = nn.ModuleList()
        for i in range(self.gnn_layers):
            if i == self.gnn_layers - 1:
                self.layers.append(GATLayer(self.dim, 1, self.gnn_dropout, self.gnn_norm, self.graph_size, self.add_edge_attr))
            else:
                self.layers.append(GATLayer(self.dim, self.gnn_heads, self.gnn_dropout, self.gnn_norm, self.graph_size, self.add_edge_attr))

    def forward(self, *argv):
        if len(argv) == 3:  
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        elif len(argv) == 2:
            x, edge_index = argv[0], argv[1]
            edge_attr = None
        else:
            raise ValueError("unmatched number of arguments.")
        
        if not (self.add_edge_attr == 'no'):
            assert edge_attr !=None
        else:
            edge_attr = None
        
        if self.gnn_residual == 'linear':
            assert hasattr(self, 'lin')
            x_in = self.lin(x)
        elif self.gnn_residual == 'simple':
            x_in = x
        else:
            x_in = 0

        for i, layer in enumerate(self.layers):
            if i != 0:
                x = self.activation(x)
            x = layer(x, edge_index, edge_attr) + x_in

        return x

    

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        if self.config.context_node:
            self.num_relations = config.num_relations + 1
        else:
            self.num_relations = config.num_relations
        self.embedding_dim = config.gnn_dim

        self.negative_adversarial_sampling = config.link_negative_adversarial_sampling
        self.adversarial_temperature = config.link_negative_adversarial_sampling_temperature
        self.reg_param = config.link_regularizer_weight

    def forward(self, embs, sample, mode='single'):
        """
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        """

        if mode == 'single':
            batch_size, negative_sample_size = sample[0].shape[0], 1

            head = embs[sample[0]].unsqueeze(1) #[n_triple, 1, dim]
            relation = self.w_relation[sample[1]].unsqueeze(1) #[n_triple, 1, dim]
            tail = embs[sample[2]].unsqueeze(1) #[n_triple, 1, dim]

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.shape

            head = embs[head_part] #[n_triple, n_neg, dim]
            relation = self.w_relation[tail_part[1]].unsqueeze(1) #[n_triple, 1, dim]
            tail = embs[tail_part[2]].unsqueeze(1) #[n_triple, 1, dim]

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.shape

            head = embs[head_part[0]].unsqueeze(1)
            relation = self.w_relation[head_part[1]].unsqueeze(1)

            tail = embs[tail_part]

        else:
            raise ValueError('mode %s not supported' % mode)

        score = self.score(head, relation, tail, mode) #[n_triple, 1 or n_neg]

        return score

    def score(self, h, r, t, mode):
        raise NotImplementedError

    def reg_loss(self):
        return torch.mean(self.w_relation.pow(2))
        # return torch.tensor(0)

    def loss(self, scores):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        positive_score, negative_score = scores
        if self.negative_adversarial_sampling:
            # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * self.adversarial_temperature, dim=1).detach()
                              * F.logsigmoid(-negative_score)).sum(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1) #[n_triple,]

        positive_score = F.logsigmoid(positive_score).squeeze(dim=1) #[n_triple,]

        assert positive_score.dim() == 1
        if len(positive_score) == 0:
            positive_sample_loss = negative_sample_loss = 0.
        else:
            positive_sample_loss = - positive_score.mean() #scalar
            negative_sample_loss = - negative_score.mean() #scalar

        loss = (positive_sample_loss + negative_sample_loss) / 2 + self.reg_param * self.reg_loss()

        return loss, positive_sample_loss, negative_sample_loss

class DistMultDecoder(Decoder):
    """DistMult score function
        Paper link: https://arxiv.org/abs/1412.6575
    """
    def __init__(self, config):
        super().__init__(config)

        print (" ### Initializing w_relation for DistMultDecoder...")
        self.register_parameter('w_relation', nn.Parameter(torch.Tensor(self.num_relations, self.embedding_dim)))

        self.embedding_range = math.sqrt(1.0 / self.embedding_dim)
        with torch.no_grad():
            self.w_relation.uniform_(-self.embedding_range, self.embedding_range)

        print (" ### DistMultDecoder Initialized")

    def score(self, head, relation, tail, mode):
        if mode == 'head-batch':
            if self.config.scaled_distmult:
                tail = tail / math.sqrt(self.embedding_dim)
            score = head * (relation * tail)
        else:
            if self.config.scaled_distmult:
                head = head / math.sqrt(self.embedding_dim)
            score = (head * relation) * tail

        score = score.sum(dim=2)
        return score

    def __repr__(self):
        return '{}(embedding_size={}, num_relations={})'.format(self.__class__.__name__,
                                                                self.embedding_dim,
                                                                self.num_relations)


    












