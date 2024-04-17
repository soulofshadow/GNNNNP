import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

from tqdm import tqdm
import json
import logging
import os


from transformers import AutoTokenizer

#load input text and label text 
#use tokenizer to input_ids

#load subgraph files
#combine them and separate as batch size

PROMPT = """{question}{options}{context}{longanswer}"""

class LLMGNP_DataLoader():
    def __init__(self, args) -> None:
        super(LLMGNP_DataLoader, self).__init__()

        self.device = args.device
        self.context_node = args.context_node
        self.context_pad = args.context_pad

        self.add_edge_attr = args.add_edge_attr

        self.num_subgraphs = args.num_subgraphs
        self.sub_graphs_choice = args.sub_graphs_choice
        self.num_nodes = args.num_nodes
        self.num_relations = args.num_relations
        self.batch_size = args.batch_size
        self.pad_node = args.pad_node

        self.model_name = args.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.max_seq_len = args.max_seq_len

        self.use_char_options_format = args.use_char_options_format
        self.prompt_context = args.prompt_context
        self.prompt_Lanswer = args.prompt_Lanswer

        relation_file = "./data/umls/relations.txt"
        id2relation = [r.strip() for r in open(relation_file)]
        self.rel_num = len(id2relation)

        relation2id = {r: i for i, r in enumerate(id2relation)}
        relation2id['master_node'] = -1
        self.id2rel_dict = {v : k for k, v in relation2id.items()}
        
        # #debug use
        if args.mode == 'test':
            self.test_dataset = self.data_process(args.t_dev_path, args.g_dev_path)
        
        else:
            # load train
            self.train_dataset = self.data_process(args.t_train_path, args.g_train_path)
            # # #load dev
            self.dev_dataset = self.data_process(args.t_dev_path, args.g_dev_path)
            #load test
            self.test_dataset = self.data_process(args.t_test_path, args.g_test_path)

    def data_process(self, text_path, graph_path):
        #text
        data_t = []
        with open(text_path, 'r', encoding='utf-8') as file:
            for line in file:
                json_obj = json.loads(line)
                data_t.append(json_obj)

        #[sample, T_data]
        lm_inputs, lm_labels = self.process_text(data_t)

        #subgraph
        data_g = []
        with open(graph_path, 'r', encoding='utf-8') as file:
            for line in file:
                json_obj = json.loads(line)
                data_g.append(json_obj)
        
        #[sample, num_choice, G_data]
        graphdata = self.process_graph(data_g)

        #batch
        batches = list(self.create_batch(lm_inputs, lm_labels, graphdata))
        return batches
    
    def process_text(self, data):

        formated = map(self.format_source, data)
        formatted_inputs, answers = zip(*formated)

        return formatted_inputs, answers
    
    def format_options(self, example, use_char_options_format: bool = False):
        if use_char_options_format:
            options_prefix = "OPTIONS:"
            separator = "."
            CHAR_OPTIONS = [f"\n{chr(x)} - " for x in range(ord("A"), ord("Z") + 1)]
            options = [a + b for a, b in zip(CHAR_OPTIONS[:len(example)],example)]
        else:
            options_prefix = "OPTIONS:\n- "
            separator = ".\n- "
            options = example

        out = options_prefix + separator.join(options) + '.'
        return out
    
    def format_source(self, source):

        #format question
        assert "question" in source
        question = "Question: " + source["question"]
            
        #format options
        if "options" in source:
            if isinstance(source["options"], dict):
                optionlist = list(source["options"].values())
            else:
                optionlist = source["options"]
            
            options = self.format_options(optionlist, self.use_char_options_format)
        else:
            options = ""
        
        #format context
        if "context" in source and self.prompt_context:
            context = "Context: " + source["context"]
        else:
            context = ""

        #format longanswer
        if "longanswer" in source and self.prompt_Lanswer:
            longanswer = "Long answer: " + source["longanswer"]
        else:
            longanswer = ""

        PROMPT = f"""{question}\n\n{options}\n\n{context}\n\n{longanswer}"""

        #Label
        if self.use_char_options_format:
            if "answer_idx" in source:
                id = source['answer_idx']
                Label = id
            elif "answer" in source:
                id = chr(optionlist.index(source["answer"]) + ord("A"))
                Label = f"{id}"
        else:
            Label = source['answer']

        return PROMPT, Label

    def process_graph(self, data):
        all_graphs = []
        for i in range(len(data)):
            one_sample_graphs = []

            if self.sub_graphs_choice == 'all':
                for graph in data[i]:
                    one_sample_graphs.append(self.process_one_graph(graph))
            elif self.sub_graphs_choice == 'solo_question':
                one_sample_graphs.append(self.process_one_graph(data[i][0]))
            elif self.sub_graphs_choice == 'solo_option':
                for graph in data[i][1:]:
                    one_sample_graphs.append(self.process_one_graph(graph))

            all_graphs.append(one_sample_graphs)

        return all_graphs

    def process_one_graph(self, data):
        
        if self.context_node:
            n_special = 1
        else:
            n_special = 0
        
        nodes = data['nodes'][:(self.num_nodes - n_special)]
        edge_list = []
        edge_attr = []

        #context
        if self.context_node:
            nodes.insert(0, self.context_pad)
            for node in nodes[1:]:
                edge_list.append([nodes[0], node])
                edge_attr.append(self.num_relations)

        for index, [a, b] in enumerate(data['edges']):
            if a in nodes and b in nodes:
                edge_list.append([a, b])
                edge_attr.append(data['edge_types'][index])

        #####originl
        source_nodes = [s for s, t in edge_list]
        target_nodes = [t for s, t in edge_list]
        node_id_to_index = {node_id: idx for idx, node_id in enumerate(nodes)}
        edge_index = torch.tensor([
            [node_id_to_index[s] for s in source_nodes],
            [node_id_to_index[t] for t in target_nodes]
        ], dtype=torch.long)

        nodes = torch.tensor(nodes, dtype=torch.long)
        if nodes.size(0) < self.num_nodes:
            pad_length = self.num_nodes - nodes.size(0)
            nodes = torch.cat([nodes, torch.tensor([self.pad_node] * pad_length, dtype=torch.long)], dim=0)

        nodes = nodes.to(self.device)
        edge_index = edge_index.to(self.device)
        nodes_mask = (nodes != self.pad_node).view(-1).int().to(self.device)

        edge_lp = torch.tensor(edge_attr).to(self.device)

        ####
        #
        if self.add_edge_attr == 'semantic':
            edge_attr = [x-self.rel_num if x > (self.rel_num - 1) else x for x in edge_attr]
            edge_attr = [self.id2rel_dict[x] for x in edge_attr]
            edge_attr = self.tokenizer(edge_attr, return_tensors='pt', max_length=12, padding="max_length", truncation=True, add_special_tokens=False).to(self.device)

            data = Data(x=nodes, edge_index=edge_index, edge_attr=edge_attr['input_ids'], mask=nodes_mask, edge_attr_mask=edge_attr['attention_mask'], edge_lp=edge_lp)
        else:
            edge_attr = torch.tensor(edge_attr).to(self.device)
            data = Data(x=nodes, edge_index=edge_index, edge_attr=edge_attr, mask=nodes_mask, edge_lp=edge_lp)

        return data


    def create_batch(self, lm_inputs, lm_labels, graphdata):
        # assert len(lmdata[0]) == len(gnndata[0])
        for i in range(0, len(lm_inputs), self.batch_size):
            lm_inputs_batch = lm_inputs[i:i + self.batch_size]
            lm_labels_batch = lm_labels[i:i + self.batch_size]

            lm_inputs_batch = self.pad_batch_lm(lm_inputs_batch)
            lm_labels_batch = self.pad_batch_lm(lm_labels_batch)

            graph_batch = graphdata[i:i + self.batch_size]
            yield tuple([lm_inputs_batch, lm_labels_batch, graph_batch])

    def pad_batch_lm(self, lm_inputs):

        max_length = self.max_seq_len - self.num_subgraphs
        encoded_inputs = self.tokenizer(lm_inputs, return_tensors='pt', padding=True, truncation=True, max_length=max_length, add_special_tokens=True).to(self.device)

        return encoded_inputs
