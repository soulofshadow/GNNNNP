
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import math
from typing import Tuple, Optional

from torch import nn
from torch_geometric.utils import degree

from utils.utils import freeze_net

def extend_mask(mask, dtype: torch.float = None):
    if mask.dim() == 3:
        extended_attention_mask = mask[:, None, :, :]
    elif mask.dim() == 2:
        extended_attention_mask = mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for attention_mask"
        )
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    return extended_attention_mask

def gelu(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)
    
activation_classes = {'gelu': GELU, 'relu': nn.ReLU, 'tanh': nn.Tanh}

class CustomizedEmbedding(nn.Module):
    def __init__(self, concept_num, 
                 concept_dim, 
                 gnn_dim,
                 pretrained_concept_emb=None, 
                 init_range=0.02, 
                 freeze_ent_emb=True, scale=1):
        super().__init__()

        self.scale = scale
        self.emb = nn.Embedding(concept_num + 1, concept_dim)
        if pretrained_concept_emb is not None:
            print (" ### Initializing embedding for CustomizedEmbedding...")
            self.emb.weight.data.fill_(0)
            self.emb.weight.data[:concept_num].copy_(pretrained_concept_emb)
        else:
            self.emb.weight.data.normal_(mean=0.0, std=init_range)

        if concept_dim != gnn_dim:
            self.cpt_transform = nn.Linear(concept_dim, gnn_dim)
            self.activation = nn.GELU()
        
        if freeze_ent_emb:
            freeze_net(self.emb)
        print (" ### CustomizedEmbedding Initialized")

    def forward(self, index):
        """
        index: size (bz, a)
        contextualized_emb: size (bz, b, emb_size) (optional)
        """
        if hasattr(self, 'cpt_transform'):
            return self.activation(self.cpt_transform(self.emb(index) * self.scale))
        return self.emb(index) * self.scale

class AveragePoolingLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, H, mask=None):

        masked_embeddings = H * mask.unsqueeze(-1)
        num_valid_nodes = mask.sum(dim=1, keepdim=True).float()

        #IF NO NODES
        all_masked_samples = (num_valid_nodes == 0)
        num_valid_nodes[all_masked_samples] = 1 

        average_pooled = masked_embeddings.sum(dim=1) / num_valid_nodes

        return average_pooled

class GraphNorm(nn.Module):
    """
        Param: []
    """
    def __init__(self, num_features, num_nodes, eps=1e-5, affine=False):
        super().__init__()
        self.eps = eps
        self.num_features = num_features
        self.num_nodes = num_nodes 
        self.affine = affine

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(self.num_features))
            self.beta = nn.Parameter(torch.zeros(self.num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def norm(self, x):
        mean = x.mean(dim = 0, keepdim = True)
        var = x.std(dim = 0, keepdim = True)
        x = (x - mean) / (var + self.eps)
        return x

    def forward(self, x):
        if x.dim() == 2:
            x_list = torch.split(x, self.num_nodes) # (tuple of [nodes, dimension], length of graph num)
        elif x.dim() == 3:
            assert x.size(1) == self.num_nodes
            x_list = x
            
        norm_list = []
        for x in x_list:
            norm_list.append(self.norm(x))
        norm_x = torch.cat(norm_list, 0)

        if self.affine:
            norm_x =  self.gamma * norm_x + self.beta

        if x.dim() == 2:
            return norm_x #[all_nodes, dim]
        elif x.dim() == 3:
            return norm_x.view(x.shape)
        
#FFN

class FFNLayer(nn.Module):
    def __init__(self, input_d, output_d, mult=4, dropout_rate=0.2, activation='gelu'):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(input_d, int(input_d * mult), bias=False),
            activation_classes[activation.lower()](),
            nn.Linear(int(input_d * mult), output_d, bias=False)
        )

    def forward(self, hidden_states):
        return self.ff(hidden_states)

#self attention layer

class SelfAttnLayer(nn.Module):
    def __init__(self, dim, self_attn_heads, dropout_self_attn):
        super().__init__()

        self.heads = self_attn_heads
        self.dim_per_head = dim // self.heads
        self.inner_dim = self.dim_per_head * self.heads
        self.dropout = dropout_self_attn

        assert self.inner_dim % self.heads == 0

        self.q = nn.Linear(dim, self.inner_dim, bias=False)
        self.k = nn.Linear(dim, self.inner_dim, bias=False)
        self.v = nn.Linear(dim, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, dim, bias=False)

    def forward(self, H, mask=None):

        # H = (b, n_node, d_node)
        # mask = (b, mask_bool)
            
        batch_size = H.size(0)

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.heads, self.dim_per_head).transpose(1, 2)
        
        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        query_states = shape(self.q(H))
        query_states = query_states * (self.dim_per_head ** -0.5)

        key_states = shape(self.k(H))
        value_states = shape(self.v(H))

        scores = torch.matmul(query_states, key_states.transpose(3, 2))

        if mask is not None:
            extended_mask = extend_mask(mask, scores.dtype)
            scores += extended_mask

        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        return attn_output
    
class SelfAttnModule(nn.Module):
    def __init__(self, dim, self_attn_layers, self_attn_heads, dropout_self_attn, pre_norm=True):
        super().__init__()

        self.layers = nn.ModuleList([SelfAttnLayer(dim, self_attn_heads, dropout_self_attn) for _ in range(self_attn_layers)])
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout_self_attn)
        self.pre_norm = pre_norm

    def forward(self, H, mask=None):

        for layer in self.layers:
            #change to pre-norm
            if self.pre_norm:
                normed_hidden_states = self.layer_norm(H)
                attention_output = layer(normed_hidden_states, mask)
                H = H + self.dropout(attention_output)

            #post-norm
            else:
                attention_output = layer(H, mask)
                forwarded_states = H + self.dropout(attention_output)
                H = self.layer_norm(forwarded_states)

        return H

# cross modality
    
class CrossModalityAttnLayer(nn.Module):
    def __init__(self, dim_q, dim_kv, cross_attn_heads, dropout_cross_attn):
        super().__init__()
        
        self.heads = cross_attn_heads
        self.dim_per_head = dim_kv // self.heads
        self.inner_dim = self.dim_per_head * self.heads
        self.dropout = dropout_cross_attn

        assert self.inner_dim % self.heads == 0
        
        self.q = nn.Linear(dim_q, self.inner_dim, bias=False)
        self.k = nn.Linear(dim_kv, self.inner_dim, bias=False)
        self.v = nn.Linear(dim_kv, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, dim_q, bias=False)

    def forward(self, H, T, mask=None):

        batch_size = H.size(0)

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.heads, self.dim_per_head).transpose(1, 2)
        
        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        query_states = shape(self.q(H))
        query_states = query_states * (self.dim_per_head ** -0.5)

        key_states = shape(self.k(T))
        value_states = shape(self.v(T))

        scores = torch.matmul(query_states, key_states.transpose(3, 2))

        if mask is not None:
            extended_mask = extend_mask(mask, scores.dtype)
            scores += extended_mask

        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        return attn_output
           
class CrossModalityAttnModule(nn.Module):
    def __init__(self, dim_q, dim_kv, cross_attn_layers, cross_attn_heads, dropout_cross_attn, pre_norm=True):
        super().__init__()

        self.layers = nn.ModuleList([CrossModalityAttnLayer(dim_q, dim_kv, cross_attn_heads, dropout_cross_attn) for _ in range(cross_attn_layers)])
        self.layer_norm = nn.LayerNorm(dim_q)
        self.dropout = nn.Dropout(dropout_cross_attn)
        self.pre_norm = pre_norm

    def forward(self, H, T, mask=None):
        for layer in self.layers:
            #change to pre-norm
            if self.pre_norm:
                normed_hidden_states = self.layer_norm(H)
                forwarded_states = layer(normed_hidden_states, T, mask)
                H = H + self.dropout(forwarded_states)

            #post-norm
            else:
                attention_output = layer(H, T, mask)
                forwarded_states = H + self.dropout(attention_output)
                H = self.layer_norm(forwarded_states)

        return H
    

#method 2
class MultiGraphAttentionWithConcat(nn.Module):
    def __init__(self, dim, dim_kv, num_graph, xattn_heads):
        super().__init__()

        self.heads = xattn_heads
        self.dim_per_head = dim_kv // self.heads
        self.inner_dim = int(self.dim_per_head * self.heads)

        assert self.inner_dim % self.heads == 0
        
        self.q = nn.Linear(dim, self.inner_dim, bias=False)
        self.k = nn.Linear(dim_kv, self.inner_dim, bias=False)
        self.v = nn.Linear(dim_kv, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, dim, bias=False)

    def forward(self, H, KV, KV_mask=None, 
                previous_kv: list = None,
                output_kv: bool = False):
        
        batch_size = H.size(0)
        n_graphs = KV.size(1)

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.heads, self.dim_per_head).transpose(1, 2)
        
        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
        
        query_states = shape(self.q(H))
        query_states = query_states * (self.dim_per_head ** -0.5)

        all_attn = []
        all_kv = []
        for i in range(n_graphs):
            if previous_kv is None:
                key_states = shape(self.k(KV[:, i]))
                value_states = shape(self.v(KV[:, i]))
            else:
                key_states, value_states = previous_kv[i]
            
            #cross-attm
            scores = torch.matmul(query_states, key_states.transpose(3, 2))
            if KV_mask is not None:
                extended_mask_i = extend_mask(KV_mask[:, i], scores.dtype)
                scores += extended_mask_i
            attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
                scores
            )  # (batch_size, n_heads, seq_length, key_length)
            attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
            attn_output = self.o(attn_output)
            all_attn.append(attn_output)
            all_kv.append((key_states, value_states,))

        concatenated_tensor = torch.cat(all_attn, dim=-1)

        if output_kv:
            return concatenated_tensor, tuple(all_kv)
        return concatenated_tensor, None

class GatedCrossAttentionModule(nn.Module):
    def __init__(self, dim, dim_kv, num_graph, xattn_heads, mult=4):
        super().__init__()

        self.layer_norm = nn.LayerNorm(dim)
        self.attn = MultiGraphAttentionWithConcat(dim, dim_kv, num_graph, xattn_heads)
        
        if num_graph >= 4:
            self.mult = 0.5
        else:
            self.mult = mult
        self.ff = FFNLayer(num_graph*dim, dim, self.mult)

        self.gate = nn.Parameter(torch.tensor([0.]))

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.Tensor]],
        nodes=None,
        nodes_mask=None,
        previous_kv: tuple = None,
        use_cache: Optional[bool] = False
    ):
        if nodes is None:
            kv = None
        else:
            normed_hidden_states = self.layer_norm(hidden_states)
            attn_out, kv = self.attn(normed_hidden_states, nodes, nodes_mask, previous_kv=previous_kv, output_kv=use_cache)
            ff_attn_out = self.ff(attn_out)

            hidden_states = hidden_states + self.gate * ff_attn_out

        return hidden_states, kv
    

