import sys
from os import path

import torch.nn as nn
import torch
from torch_geometric.data import Batch

from itertools import chain

from modeling import modeling_gnn
from modeling import modeling_gnp
from modeling import t5
from utils import layers
from utils import utils

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from transformers import T5ForConditionalGeneration, T5Config

class LLMGNP(nn.Module):

    '''
    text inputs [ batch , ]
    graph [batch , choices]
    '''
    def __init__(self, config):
        super(LLMGNP, self).__init__()
        self.config = config

        #LLM
        print ("Initializing LLM...")

        if config.cross_gnp:
            #configuration
            custom_config, kwargs = AutoConfig.from_pretrained(
                    config.model_name,
                    return_unused_kwargs=True)
            #add
            custom_config.cross_gnp = self.config.cross_gnp
            
            custom_config.num_graph = self.config.num_subgraphs
            custom_config.xattn_heads = self.config.xattn_heads
            custom_config.xattn_after = self.config.xattn_after
            custom_config.xattn_mult = self.config.ffn_mult
            custom_config.dim_graph = self.config.gnn_dim

            self.llm_tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self.llm_model = t5.T5ForConditionalGeneration.from_pretrained(config.model_name, config=custom_config)
        else:
            self.llm_tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self.llm_model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)

        self.config.llm_dim = self.llm_model.model_dim

        print ("LLM Initialized")

        #GNP
        if config.add_gnp or config.cross_gnp:
            print ("Initializing GNP module...")
            self.GNP = modeling_gnp.GNP(self.config)
            print ("GNP Initialized")

        #others
        if config.llm_frozen:
            utils.freeze_net(self.llm_model)
            #unfreeze xattn]
            if config.cross_gnp:
                for block in self.llm_model.decoder.block:
                    if hasattr(block, 'fusion'):
                        utils.unfreeze_net(block.fusion)
     
    def get_embedding(self, lminputs, gnndata):

        all_hidden_states = None

        if self.config.cross_gnp:
            kwargs = {}
            kwargs['output_hidden_states'] = True
            kwargs['return_dict'] = True
            all_hidden_states = self.llm_model.encoder(lminputs['input_ids'], lminputs['attention_mask'], **kwargs).hidden_states
        
        if self.config.add_edge_attr == 'semantic':
            for persample in gnndata:
                for persubgraph in persample:
                    persubgraph.edge_attr = self.llm_model.encoder(input_ids=persubgraph.edge_attr, attention_mask=persubgraph.edge_attr_mask).last_hidden_state.mean(dim=1).squeeze()

        return self.llm_model.shared(lminputs['input_ids']), all_hidden_states

    def forward(self, batch, flag='train'):

        lminputs, lmlabels, gnndata = batch
        
        T, all_hidden_states = self.get_embedding(lminputs, gnndata)
        T_mask = lminputs['attention_mask']

        #use GNP
        if self.config.add_gnp or self.config.cross_gnp:
            gnp, lp_loss, cross_nodes, cross_nodes_mask = self.GNP(T, all_hidden_states, T_mask, gnndata)
        else:
            lp_loss = (0., 0., 0.)

        #put for GNP
        if self.config.add_gnp:
            assert gnp.size(0) == T.size(0) * self.config.num_subgraphs
            gnp = gnp.view(T.size(0), self.config.num_subgraphs, self.config.llm_dim)

            #prepend gnp to token embedding
            inputs_embeds = torch.cat([gnp, T], dim=1)
            mask_pad = torch.full((T.size(0), self.config.num_subgraphs), 1, dtype=torch.long).to(self.config.device)
            inputs_masks = torch.cat([mask_pad, T_mask], dim=1)

            labels = lmlabels['input_ids']
            labels[labels == self.llm_tokenizer.pad_token_id] = -100
            labels_masks = lmlabels['attention_mask']
        else:
            inputs_embeds = T
            inputs_masks = T_mask
            
            labels = lmlabels['input_ids']
            labels[labels == self.llm_tokenizer.pad_token_id] = -100
            labels_masks = lmlabels['attention_mask']

        #
        if self.config.cross_gnp:
            cross_nodes = cross_nodes.view(T.size(0), self.config.num_subgraphs, self.config.num_nodes, -1)
            cross_nodes_mask = cross_nodes_mask.view(T.size(0), self.config.num_subgraphs, -1)
        else:
            cross_nodes = None
            cross_nodes_mask = None
            
        #Train
        #LLM output
        if flag == 'train':
            if self.config.cross_gnp:
                output = self.llm_model(inputs_embeds = inputs_embeds,
                                    attention_mask = inputs_masks,
                                    labels = labels,
                                    decoder_attention_mask=labels_masks,
                                    nodes=cross_nodes,
                                    nodes_mask=cross_nodes_mask
                                    )
            else:
                output = self.llm_model(inputs_embeds = inputs_embeds,
                                        attention_mask = inputs_masks,
                                        labels = labels,
                                        decoder_attention_mask=labels_masks
                                        )

            logits = output.logits
            lm_loss = output.loss

            return logits, lm_loss, lp_loss

        #Evl
        #Generate
        if flag == 'eval':
            if self.config.cross_gnp:
                output = self.llm_model.generate(inputs_embeds=inputs_embeds, 
                                    attention_mask=inputs_masks,
                                    max_length=self.config.max_tag_len,
                                    nodes=cross_nodes,
                                    nodes_mask=cross_nodes_mask)
            else:
                output = self.llm_model.generate(inputs_embeds=inputs_embeds, 
                                        attention_mask=inputs_masks,
                                        max_length=self.config.max_tag_len)
                                
            return output
        