import sys
from os import path

import torch.nn as nn
import torch

from torch_geometric.data import Batch
from itertools import chain

from modeling import modeling_gnn
from utils import layers
from utils import utils
from utils.layers import GraphNorm, GELU

class GNP(nn.Module):
    def __init__(self, config):
        super(GNP, self).__init__()

        self.config = config

        #1.first embed subgraph with initial embedding
        self.ent_embed_init = layers.CustomizedEmbedding(config.concept_num, 
                                                         config.concept_dim, 
                                                         config.gnn_dim,
                                                         config.pretrained_concept_emb, 
                                                         config.init_range, config.freeze_ent_emb)
        self.dropout_emb = nn.Dropout(config.dropout_emb)

        if self.config.add_edge_attr == 'semantic':
            self.edge_encoder = layers.FFNLayer(config.llm_dim, 
                                            config.gnn_dim, 
                                            config.ffn_mult, 
                                            config.dropout_ffn, 
                                            config.activation)

        #2. GNP
        if self.config.add_gnp:
            # GNP
            self.gnn_block = GNNBlock(self.config)
            # Average Pool    
            self.avgpool = layers.AveragePoolingLayer()
            # another FFN to map Graph to dimension of LLM
            self.fc_gnn2llm = layers.FFNLayer(config.gnn_dim, config.llm_dim,
                                            config.ffn_mult, config.dropout_ffn, config.activation)
        
        #3. extra for only decoder
        #GNP block with token_embed and Graph
        if self.config.cross_gnp:
            #GNPCross
            self.gnn_blocks = nn.ModuleList([GNNCrossBlock(self.config, id=i) for i in range(self.config.cross_gnp_num)])

            if self.config.cross_gnp_choice == 3:
                self.Vh = nn.Linear(config.gnn_dim, config.gnn_dim)
                self.Vx = nn.Linear(config.gnn_dim, config.gnn_dim)
                self.activation_residual = nn.Sequential(
                    nn.GELU(),
                    nn.Dropout(0.1))
                self.activation_gat = nn.Sequential(
                    nn.GELU(),
                    nn.Dropout(0.1))

        #4. Link Prediction   
        if config.is_lp:
            self.linkpred = modeling_gnn.DistMultDecoder(config)


    def forward(self, token_embed, lm_hidden_states, token_mask, subgraphs):
        #it should be
        '''
        real batch_size = bs * num_choices
        text: [batch, token, D]
        subgraphs: [batch, G]

        gnp: token_embed, token_mask, subgraphs, graph_mask
        cross: lm_hidden_states, token_mask, subgraphs
        '''
        batch_num = token_embed.size(0) * self.config.num_subgraphs #batch_size * num_subgraphs
        
        #init H
        H = self.batch_choice_of_graph(subgraphs)
        H.x = self.ent_embed_init(H.x)
        H.x = self.dropout_emb(H.x)
        H_mask = H.mask
        H_mask = H_mask.view(batch_num, self.config.num_nodes)

        if self.config.add_edge_attr == 'semantic':
            assert hasattr(self, 'edge_encoder')
            H.edge_attr = self.edge_encoder(H.edge_attr)

        #GNP blocks 
        output = None
        if self.config.add_gnp: 
            #make sure no use context node
            #GNP module
            H_node_embed = self.gnn_block(H, token_embed, H_mask, token_mask)
            #
            H_node_final = H_node_embed.view(batch_num, self.config.num_nodes, -1) #save for LP
            #Avg Pool
            H_graph = self.avgpool(H_node_final, H_mask)
            #Domain Projector
            output = self.fc_gnn2llm(H_graph)


        #decoder cross
        nodes = None
        nodes_mask = None
        # Design choice: use lask K hidden_states to query GNN
        # or to use every x hidden_states to query GNN
        # Currently, I chose the former. As deeper LM features contain more semantic information,
        # while the shallower LM features contain more linguistic information
        #remove lm_hidden_states[-1] cause it is just normalized of lm_hidden_states[-2]
        if self.config.cross_gnp:
            #Cross module
            nodes_input = H.x.view(batch_num, self.config.num_nodes, -1)
            for gnn_block, lm_hidden_states in zip(self.gnn_blocks, lm_hidden_states[-len(self.gnn_blocks):]):
                H.x = gnn_block(H, lm_hidden_states, token_mask)

            if self.config.cross_gnp_choice == 1:
                nodes_after_cross = H.x.view(batch_num, self.config.num_nodes, -1)
                H_node_final = nodes_after_cross #save for LP
                nodes = nodes_after_cross
                nodes_mask = H_mask

            if self.config.cross_gnp_choice == 3:
                nodes_after_cross = H.x.view(batch_num, self.config.num_nodes, -1)
                nodes_after_cross = self.activation_gat(nodes_after_cross)
                nodes_after_cross = self.activation_residual(self.Vh(nodes_input) + self.Vx(nodes_after_cross))
                H_node_final = nodes_after_cross #save for LP
                nodes = nodes_after_cross
                nodes_mask = H_mask

        '''
        LP
        '''
        if self.config.is_lp:
            pos_triples, neg_nodes = self.create_pos_neg_trip(H)

            pos_samples = pos_triples #[3, `total_n_triple`]

            _n_neg = neg_nodes.size(1)
            head_negative_sample = neg_nodes[:, :_n_neg//2]             #[`total_n_triple`, n_neg//2]
            tail_negative_sample = neg_nodes[:, _n_neg//2:_n_neg//2*2] 

            embs = H_node_final.view(-1, self.config.gnn_dim)

            positive_score  = self.linkpred(embs, pos_samples) #[`total_n_triple`, 1]
            head_neg_scores = self.linkpred(embs, (pos_samples, head_negative_sample), mode='head-batch')
            tail_neg_scores = self.linkpred(embs, (pos_samples, tail_negative_sample), mode='tail-batch')
            negative_score = torch.cat([head_neg_scores, tail_neg_scores], dim=-1) #[`total_n_triple`, total_n_neg]
            scores = (positive_score, negative_score)
            link_loss, pos_link_loss, neg_link_loss = self.linkpred.loss(scores)
        else:
            link_loss = pos_link_loss = neg_link_loss = 0.


        return output, (link_loss, pos_link_loss, neg_link_loss), nodes, nodes_mask
    

    def batch_choice_of_graph(self, subgraphs):
        '''
        graph [batch , choices] -> [batch* choices]
        '''
        #get x embedding initialized
        Graphs = list(chain.from_iterable(subgraphs))
        big = Batch.from_data_list(Graphs)
        big.x[big.x == self.config.pad_node] = self.config.concept_num
        if self.config.context_node:
            big.x[big.x == self.config.context_pad] = self.config.concept_num
        big.x = big.x.to(torch.long)
        return big

    def create_pos_neg_trip(self, H):

        E = len(H.edge_index[0])
        if E == 0:
            # print ('KG with 0 node', file=sys.stderr)
            effective_num_nodes = 1
        else:
            effective_num_nodes = int(H.edge_index.max()) + 1

        positions = torch.arange(E).to(self.config.device)
        #remove context
        if self.config.context_node:
            positions = positions[H.edge_lp != self.config.num_relations]

        drop_count = int(len(positions) * self.config.link_drop_probability)
        if len(positions) > 0 and drop_count > 0:
            drop_idxs = torch.multinomial(torch.full((len(positions),), 1.0), drop_count, replacement=False).to(self.config.device)
        else:
            drop_idxs = torch.tensor([]).long().to(self.config.device)
        drop_positions = positions[drop_idxs] #[drop_count, ]

        mask = torch.zeros((E,)).long().to(self.config.device)
        mask = mask.index_fill_(dim=0, index=drop_positions, value=1).bool()

        pos_edge_index = H.edge_index[:, mask].long()
        pos_edge_type  = H.edge_lp[mask].long()

        pos_triples = [pos_edge_index[0], pos_edge_type, pos_edge_index[1]]

        num_edges = len(pos_edge_type)
        num_corruption = self.config.link_negative_sample_size

        neg_nodes = torch.randint(0, effective_num_nodes, (num_edges, num_corruption)).to(self.config.device)

        return pos_triples, neg_nodes
    

class GNNBlock(nn.Module):
    def __init__(self, config):
        super(GNNBlock, self).__init__()
        self.config = config

        # gnn block contains,
        #a Transfer(option), a GAT, a Self_attn, a Cross_attn

        #cause T5 middle hidden states contain unnormalized feature
        #it is choosing to transfer or not ( cause Act will lose information)
        self.fc_llm2gnn = layers.FFNLayer(config.llm_dim, 
                                        config.gnn_dim, 
                                        config.ffn_mult, 
                                        config.dropout_ffn, 
                                        config.activation)
    
        #
        self.GNN = modeling_gnn.GATNet(config.gnn_dim,
                                       config.gnn_layers,
                                       config.gnn_heads,
                                       config.num_nodes,
                                       config.gnn_norm,
                                       config.gnn_residual,
                                       config.add_edge_attr,
                                       config.dropout_gnn,
                                       config.activation)
        
        self.self_attn = layers.SelfAttnModule(config.gnn_dim,
                                               config.self_attn_layers,
                                               config.self_attn_heads,
                                               config.dropout_self_attn,
                                               config.pre_norm)
        
        self.cross_attn = layers.CrossModalityAttnModule(config.gnn_dim,
                                                         config.gnn_dim,
                                                         config.cross_attn_layers,
                                                         config.cross_attn_heads,
                                                         config.dropout_cross_attn,
                                                         config.pre_norm)
        

    def forward(self, H, T, H_mask, T_mask):

        origin_size = H.x.size(0)

        #FFN
        T = self.fc_llm2gnn(T)
        T = torch.repeat_interleave(T, repeats=self.config.num_subgraphs, dim=0)
        T_mask = torch.repeat_interleave(T_mask, repeats=self.config.num_subgraphs, dim=0)

        #GAT
        H = self.GNN(H)
        #reshape for Attn
        H = H.view(T.size(0), self.config.num_nodes, -1)

        #ATTN
        H = self.self_attn(H, H_mask)
        H = self.cross_attn(H, T, T_mask)

        return H.view(origin_size, -1)
  

class GNNCrossBlock(nn.Module):
    def __init__(self, config, id):
        super(GNNCrossBlock, self).__init__()

        self.id = id
        self.config = config

        gnn_layers = 1
        cross_attn_layers = 1
        ffn_mult = 1
        
        self.GNN = modeling_gnn.GATNet(config.gnn_dim,
                                       gnn_layers,
                                       config.gnn_heads,
                                       config.num_nodes,
                                       config.gnn_norm,
                                       config.gnn_residual,
                                       config.add_edge_attr,
                                       config.dropout_gnn,
                                       config.activation)
        
        self.cross_attn = layers.CrossModalityAttnModule(config.gnn_dim,
                                                         config.llm_dim,
                                                         cross_attn_layers,
                                                         config.cross_attn_heads,
                                                         config.dropout_cross_attn,
                                                         config.pre_norm)
            
        if self.config.cross_gnp_choice == 1:
            self.norm = nn.LayerNorm(config.llm_dim)
        elif self.config.cross_gnp_choice == 3:
            self.readout = nn.Sequential(
                    nn.Linear(config.gnn_dim, config.gnn_dim),
                    nn.LayerNorm(config.gnn_dim),
                    GELU(),
                    nn.Linear(config.gnn_dim, config.gnn_dim)
                )
            self.merge_ffn = layers.FFNLayer(config.gnn_dim*2, 
                                            config.gnn_dim, 
                                            ffn_mult, 
                                            config.dropout_ffn, 
                                            config.activation)


    def forward(self, H, T, T_mask):

        origin_size = H.x.size(0)

        #Text
        T = torch.repeat_interleave(T, repeats=self.config.num_subgraphs, dim=0)
        T_mask = torch.repeat_interleave(T_mask, repeats=self.config.num_subgraphs, dim=0)
        #GAT
        H = self.GNN(H)
        #reshape for Attn
        H = H.view(T.size(0), self.config.num_nodes, -1)

        # Implement better pooling over the sentence embedding
        # Design choices:
        # 1.- use max | mean pooling (no use of context node) similar to GNP
        if self.config.cross_gnp_choice == 1:
            if id != self.config.cross_gnp_num:
                T = self.norm(T)
            H = self.cross_attn(H, T, T_mask)
        # 2.- use multi_head self attention as pooling
        # 3.✓ use multi_head cross attention pooling (where the query is the gnn context node)
        if self.config.cross_gnp_choice == 3:
            H = self.readout(H)
            context_node_gnn_feats = H[:, 0, :].clone() # [bs, node_dim]
            context_node_lm_feats = self.cross_attn(context_node_gnn_feats.unsqueeze(1), 
                                                    T, 
                                                    T_mask)
            context_node_lm_feats = context_node_lm_feats.squeeze(1) # [bs, node_dim]
            context_node_feats = torch.cat([context_node_lm_feats, context_node_gnn_feats], dim=1)
            context_node_feats = self.merge_ffn(context_node_feats)
            # residual link
            context_node_feats = context_node_feats + context_node_gnn_feats
            H[:, 0, :] = context_node_feats


        return H.view(origin_size, -1)



        


