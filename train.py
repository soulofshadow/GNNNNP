import sys
import argparse
import logging
import os
import time
from os import path
import json

# 
from utils.log_helper import logger_init
from modeling.modeling_llm import LLMGNP
from utils.data_util import LLMGNP_DataLoader
from utils import utils

from transformers import AdamW
from torch.optim import SGD, Adam, RAdam

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm, trange
import pandas as pd

# 在使用 tokenizers 之前禁用并行处理
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

OPTIMIZER_CLASSES = {
    'sgd': SGD,
    'adam': Adam,
    'adamw': AdamW,
    'radam': RAdam,
}
# 
from torchinfo import summary

try:
    from transformers import (ConstantLRSchedule, WarmupLinearSchedule, WarmupConstantSchedule)
except:
    from transformers import get_constant_schedule, get_constant_schedule_with_warmup,  get_linear_schedule_with_warmup


class ModelConfig:
    def __init__(self, args):

        self.__dict__.update(args)

        #extend
        self.project_dir = os.getcwd()
        self.dataset_dir = os.path.join(self.project_dir, self.data_dir)
        self.dataset_name = self.dataset
        #text
        self.t_train_path = os.path.join(self.dataset_dir, self.dataset_name, 'raw', 'train.jsonl')
        self.t_dev_path = os.path.join(self.dataset_dir, self.dataset_name, 'raw', 'dev.jsonl')
        self.t_test_path = os.path.join(self.dataset_dir, self.dataset_name, 'raw', 'test.jsonl')
        #subgraph
        self.g_train_path = os.path.join(self.dataset_dir, self.dataset_name, 'subgraphed', 'train.jsonl')
        self.g_dev_path = os.path.join(self.dataset_dir, self.dataset_name, 'subgraphed', 'dev.jsonl')
        self.g_test_path = os.path.join(self.dataset_dir, self.dataset_name, 'subgraphed', 'test.jsonl')
        #entity embedding
        self.emb_ent_path = os.path.join(self.dataset_dir, self.ent_emb_paths)

        #init embedding for nodes
        cp_emb = [np.load(self.emb_ent_path)]
        cp_emb = np.concatenate(cp_emb, 1)
        cp_emb = torch.tensor(cp_emb, dtype=torch.float)
        concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)

        print('| num_concepts: {} |'.format(concept_num))

        if self.random_ent_emb:
            cp_emb = None
            self.freeze_ent_emb = False
            concept_dim = self.gnn_dim
        else:
            self.freeze_ent_emb = self.freeze_ent_emb

        self.concept_num = concept_num
        self.concept_dim = concept_dim
        self.pretrained_concept_emb = cp_emb

        #output options not full answer string  
        if self.use_char_options_format:
            self.max_tag_len = 4

        #subgraph num set
        if self.sub_graphs_choice == 'solo_question':
            self.num_subgraphs = 1
        elif self.sub_graphs_choice == 'solo_option':
            assert self.dataset_name == 'medqa'
            self.num_subgraphs = 4
        elif self.sub_graphs_choice == 'all':
            if self.dataset_name == 'medqa':
                self.num_subgraphs = 5
            else:
                self.num_subgraphs = 1
        #
        if self.cross_gnp_choice == 3:
            self.context_node = True

        #get relation num
        file_dir = f"{os.sep}".join(self.emb_ent_path.split(os.sep)[:-1]) 
        relation_path = os.path.join(file_dir, 'relations.txt')
        id2relation = [r.strip() for r in open(relation_path)]
        self.num_relations = len(id2relation) * 2

        utils.check_path(self.save_dir)

        ###log
        self.logs_save_dir = os.path.join(self.project_dir, 'logs')
        if self.add_gnp and self.cross_gnp:
            log_file_name = 'GNP+CrossGNP'
        elif self.add_gnp:
            log_file_name = 'GNP'
        elif self.cross_gnp:
            log_file_name = 'CrossGNP'
        else:
            log_file_name = 'LLM'
        log_file_name = log_file_name + '_'  + self.sub_graphs_choice + '_' + self.dataset_name
        self.log_file_name = log_file_name
        logger_init(log_file_name=log_file_name, log_level=logging.INFO,
                    log_dir=self.logs_save_dir)
        logging.info(" ### Right now config ")
        for key, value in self.__dict__.items():
            if key == 'pretrained_concept_emb':
                logging.info(f"### {key} = {value.size()}")
            else:
                logging.info(f"### {key} = {value}")

def load_dataset(config):

    dataset = LLMGNP_DataLoader(config)

    print('-' * 71)
    print("dataset loaded: ")
    if config.mode == 'test':
        print("test dataset size:", len(dataset.test_dataset) * config.batch_size)
    else:
        print("train dataset size:", len(dataset.train_dataset) * config.batch_size)
        print("dev dataset size:", len(dataset.dev_dataset) * config.batch_size)
        print("test dataset size:", len(dataset.test_dataset) * config.batch_size)
    print('-' * 71)

    return dataset

def construct_model(config):

    freeze_flag = "Frozen" if config.llm_frozen else "Unfrozen"
    addgnp_flag = "With GNP" if config.add_gnp else "Without GNP"

    print('-' * 71)
    print("constructing model:")
    print("model name:", config.model_name)
    print("model type:", "LLM" + " " + freeze_flag + " " + addgnp_flag)
    model = LLMGNP(config)
    print('-' * 71)

    return model

def train(config):

    dataset = load_dataset(config)
    model = construct_model(config)

    tokenizer = model.llm_tokenizer


    #########################################################
    # Load from resume
    #########################################################
    if config.resume:
        print("loading from checkpoint: {}".format(config.resume_checkpoint))
        checkpoint = torch.load(config.resume_checkpoint, map_location='cpu')
        last_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        model.load_state_dict(checkpoint["model"], strict=False)
        
        best_dev_epoch = checkpoint["best_dev_epoch"]
        best_dev_acc = checkpoint["best_dev_acc"]
        print(f"resume from global_step {global_step}, last_epoch {last_epoch}")

        config.save_dir = os.path.dirname(config.resume_checkpoint)
    else:
        last_epoch = -1
        global_step = 0
        best_dev_epoch = best_dev_acc = 0

    if config.load_model_path and config.load_model_path not in ["None", None]:
        print(f'loading and initializing model from {config.load_model_path}')
        checkpoint = torch.load(config.load_model_path, map_location='cpu')

        model_state_dict = checkpoint["model"]
        model.load_state_dict(model_state_dict, strict=False)
        print(f'success load from {config.load_model_path}')

    model_file_name = config.log_file_name + '_model.pt'
    model_path = os.path.join(config.save_dir, model_file_name)

    #########################################################
    # Check number of Params
    #########################################################

    summaries = summary(model)
    print('-' * 71)
    print("all parameters: ")
    # print(summaries)
    logging.info(f" ### Total params: {formatted_number(summaries.total_params)} ")
    logging.info(f" ### Trainable params:: {formatted_number(summaries.trainable_params)} ")
    logging.info(f" ### Non-trainable params:: {formatted_number(summaries.total_params - summaries.trainable_params)} ")
    print('-' * 71)
    
    #########################################################
    # Create an optimizer
    #########################################################

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
            "lr": config.learning_rate
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": config.learning_rate
        },
    ]
    optimizer = OPTIMIZER_CLASSES[config.optim](optimizer_grouped_parameters)
    if config.resume:
        optimizer.load_state_dict(checkpoint["optimizer"])
    

    #########################################################
    # Create a scheduler
    #########################################################

    num_training_steps = len(dataset.train_dataset) * config.epochs
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)


    if args.lr_schedule == 'fixed':
        try:
            scheduler = ConstantLRSchedule(optimizer)
        except:
            scheduler = get_constant_schedule(optimizer)
    elif args.lr_schedule == 'warmup_constant':
        try:
            scheduler = WarmupConstantSchedule(optimizer, warmup_steps=num_warmup_steps, last_epoch=last_epoch)
        except:
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, last_epoch=last_epoch)
    elif args.lr_schedule == 'warmup_linear':
        try:
            WarmupConstantSchedule()
            scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=num_warmup_steps, last_epoch=last_epoch)
        except:
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, last_epoch=last_epoch)
    if config.resume:
        scheduler.load_state_dict(checkpoint["scheduler"])


    print('-' * 71)
    print("optimizer and scheduler: ")
    print(optimizer)
    print(scheduler)
    print('-' * 71)

    #############################################################
    #   Training
    #############################################################

    model.to(config.device)

    print()
    print('-' * 71)

    print ('llm_task', config.llm_task, 'lp_task', config.lp_task)

    total_loss_acm = llm_loss_acm =  0.0
    link_loss_acm = pos_link_loss_acm = neg_link_loss_acm = 0.0
    total_time = 0
    n_batches_count = 0

    #
    best_test_result = 0
    best_test_epochs = 0
    #

    model.train()

    for epoch_id in trange(0, config.epochs, desc="Epoch"):
        if last_epoch + 1 > epoch_id:
            time.sleep(1)
            continue
        
        logging.info(f" ### Training epoch: {epoch_id} ")
        model.train()
        for i, batch in tqdm(enumerate(dataset.train_dataset), desc="Batch"):
            start_time = time.time()
            optimizer.zero_grad()

            logits, lm_loss, lp_loss = model(batch, 'train')
            link_loss, pos_link_loss, neg_link_loss = lp_loss

            loss = config.llm_task * lm_loss + config.lp_task * link_loss

            total_loss_acm += float(loss)
            n_batches_count += 1
            llm_loss_acm += float(lm_loss)
            link_loss_acm += float(link_loss)
            pos_link_loss_acm += float(pos_link_loss)
            neg_link_loss_acm += float(neg_link_loss)

            loss.backward()

            if config.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            if loss != loss:
                # print grad check
                v_n = []
                v_v = []
                v_g = []
                for name, parameter in model.named_parameters():
                    v_n.append(name)
                    v_v.append(parameter.detach().cpu().numpy() if parameter is not None else [0])
                    v_g.append(parameter.grad.detach().cpu().numpy() if parameter.grad is not None else [0])
                for i in range(len(v_n)):
                    if np.isnan(np.max(v_v[i]).item()) or np.isnan(np.min(v_v[i]).item()):
                        color = '\033[31m' + '\033[1m' + '*'
                    else:
                        color = '\033[92m' + '\033[1m' + ' '
                    print('%svalue %s: %.3e ~ %.3e' % (color, v_n[i], np.min(v_v[i]).item(), np.max(v_v[i]).item()))
                    print('%sgrad  %s: %.3e ~ %.3e' % (color, v_n[i], np.min(v_g[i]).item(), np.max(v_g[i]).item()))

                print(f"LM_loss: {lm_loss}")
                print(f"LP_loss: {lp_loss}")
                raise Exception('NaN in loss, crack!')
            
            optimizer.step()
            scheduler.step()

            total_time += (time.time() - start_time)

            if (global_step + 1) % config.log_interval == 0:
                ms_per_batch = 1000 * total_time / config.log_interval
                print('| step {:5} |  lr: {:9.7f} | total loss {:7.4f} | ms/batch {:7.2f} |'.format(global_step, 
                                                                                                    scheduler.get_last_lr()[0], 
                                                                                                    total_loss_acm/n_batches_count, 
                                                                                                    ms_per_batch))
                logging.info("| step: {:5} \
                              | lr: {:9.7f} \
                              | total loss: {:7.4f} \
                              | ms/batch: {:7.2f} |".format(global_step, 
                                                            scheduler.get_last_lr()[0], 
                                                            total_loss_acm/n_batches_count, 
                                                            ms_per_batch))

                total_loss_acm = llm_loss_acm =  0.0
                link_loss_acm = pos_link_loss_acm = neg_link_loss_acm = 0.0
                total_time = 0
                n_batches_count = 0

            global_step += 1 # Number of batches processed up to now

        #every epoch
        #############################################################
        #   Validate
        #############################################################
        model.eval()
        accuracy, precision, recall, f1 = validate(model, tokenizer, dataset.dev_dataset, config.use_char_options_format)
        
        logging.info(f" ### Eval epoch: {epoch_id} ")
        print ('epoch:', epoch_id, 'dev_acc:', accuracy)
        logging.info("| epoch: {:3} \
                      | step: {:5} \
                      | dev_acc: {:7.4f} \
                      | dev_precision: {:7.4f} \
                      | dev_recall: {:7.4f} \
                      | dev_f1: {:7.4f} |".format(epoch_id, 
                                                  global_step,
                                                  accuracy,
                                                  precision,
                                                  recall,
                                                  f1))
        
        ###
        accuracy_t, precision_t, recall_t, f1_t = validate(model, tokenizer, dataset.test_dataset, config.use_char_options_format)
        logging.info(f" ### Test epoch: {epoch_id} ")
        logging.info("| epoch: {:3} \
                      | step: {:5} \
                      | test_acc: {:7.4f} \
                      | test_precision: {:7.4f} \
                      | test_recall: {:7.4f} \
                      | test_f1: {:7.4f} |".format(epoch_id, 
                                                  global_step,
                                                  accuracy_t,
                                                  precision_t,
                                                  recall_t,
                                                  f1_t))
        if accuracy_t > best_test_result:
            best_test_result = accuracy_t
            best_test_epochs = epoch_id
        ###

        if accuracy >= best_dev_acc:
            best_dev_acc = accuracy
            best_dev_epoch = epoch_id

        logging.info(f" ### Best eval epoch: {best_dev_epoch} , Best eval acc: {best_dev_acc}")

        # Save the model checkpoint
        if (config.save_model==2) or (best_test_epochs == epoch_id):
            #delete check 
            model_files = os.listdir(config.save_dir)
            for model_file in model_files:
                if config.log_file_name in model_file:
                    model_file_path = os.path.join(config.save_dir, model_file)
                    os.remove(model_file_path)

            model_state_dict = model.state_dict()
            try:
                del model_state_dict["GNP.ent_embed_init.emb.weight"]
            except:
                pass
            checkpoint = {"model": model_state_dict, 
                          "optimizer": optimizer.state_dict(), 
                          "scheduler": scheduler.state_dict(), 
                          "epoch": epoch_id, 
                          "global_step": global_step, 
                          "best_dev_epoch": best_dev_epoch, 
                          "best_dev_acc": best_dev_acc, 
                          "config": config}
            
            
            torch.save(checkpoint, model_path + ".{}".format(epoch_id))


    #test result
    logging.info(f" ### Best test epoch: {best_test_epochs} , Best test acc: {best_test_result}")

def validate(model, tokenizer, dataset, is_target, output_preds=False):
    model.eval()

    predictions = []
    references = []

    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataset), desc="Batch"):
            #ground truth
            labels = batch[1]['input_ids']
            labels[labels == -100] = 0
            refs = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            #preds
            output = model(batch, 'eval')   
            preds = tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            #postprocess
            preds = extract_letters(preds, is_target)
            refs = extract_letters(refs, is_target)

            predictions.extend(preds)
            references.extend(refs)

    # 计算各种指标
    accuracy = accuracy_score(references, predictions)
    precision = precision_score(references, predictions, average='macro', zero_division=0)  # 'macro' 未加权平均
    recall = recall_score(references, predictions, average='macro', zero_division=0)
    f1 = f1_score(references, predictions, average='macro')

    if output_preds:
        jsonl_filename = 'output.jsonl'
        output_path = os.path.join(model.config.logs_save_dir, model.config.log_file_name + '_' + jsonl_filename)
        # 将数据组合成字典列表
        data_list = [{'predicted_text': predicted, 'label': label} for predicted, label in zip(predictions, references)]
        # 将字典列表写入JSONL文件
        with open(output_path, 'w', encoding='utf-8') as jsonl_file:
            for data in data_list:
                jsonl_file.write(json.dumps(data, ensure_ascii=False) + '\n')
        print(f'Data has been written to {output_path}')

    return accuracy, precision, recall, f1

def test(config):
    # assert config.load_model_path is not None
    logging.info(f" ### Test Result")

    #dataset
    dataset = load_dataset(config).test_dataset

    #model
    model = construct_model(config)

    if config.load_model_path and config.load_model_path not in ["None", None]:
        print (f'loading and initializing model from {config.load_model_path}')
        checkpoint = torch.load(config.load_model_path, map_location='cpu')

        model_state_dict = checkpoint["model"]
        model.load_state_dict(model_state_dict, strict=False)
        print(f'success load from {config.load_model_path}')

    model.to(config.device)
    model.eval()
    summaries = summary(model)

    accuracy, precision, recall, f1 = validate(model, model.llm_tokenizer, dataset, is_target=config.use_char_options_format, output_preds=True)
    print('-' * 71)
    print ('test_acc', accuracy)
    print('-' * 71)

    logging.info("| test_acc: {:7.4f} \
                  | test_precision: {:7.4f} \
                  | test_recall: {:7.4f} \
                  | test_f1: {:7.4f} |".format(accuracy,
                                            precision,
                                            recall,
                                            f1))
    

    # model_file_name = config.log_file_name + '_model_scripted.pt'
    # model_path = os.path.join(config.save_dir, model_file_name)

    # model_scripted = torch.jit.script(model) # Export to TorchScript
    # model_scripted.save(model_path) # Save


def formatted_number(value):
    return '{:,}'.format(value)
def format_approximate_value(value):
    if value >= 1e9:
        return '{:.1f}B'.format(value / 1e9)
    elif value >= 1e6:
        return '{:.1f}M'.format(value / 1e6)
    else:
        return str(value)
    
import re
def get_first_letter(input_str):
    match = re.search(r'[a-zA-Z]', input_str)
    if match:
        return match.group()
    else:
        return 'Unknown'

def extract_letters(labels, is_target: bool = False):
    extracted_letters = []

    for item in labels:
        answer = item.strip()
        if answer and answer[-1] in [".", ",", "?", " ", "\n"]:
            answer = answer[:-1].strip()  

        if is_target:
            if len(answer) == 1 and answer in 'ABCDEFG':
                extracted_letters.append(answer.upper())
            # corner case 2: target = (B), prediction = B.
            elif answer[0] == "(" and answer[-1] == ")":
                answer = answer[1:-1].strip()
                extracted_letters.append(answer.upper())
            else:
                first_letter = get_first_letter(answer)
                extracted_letters.append(first_letter)

        else:
            answer = answer.split("answer is")[-1].strip()
            answer = answer.split("final answer")[-1].strip()
            answer = answer.split("Final answer")[-1].strip()
            answer = answer.split("answer:")[-1].strip()
            answer = answer.split("Answer:")[-1].strip()
            if answer and answer[0] in [".", ",", "?", " ", "\n", ":"]:
                answer = answer[1:].strip()
            if answer and answer[-1] in [".", ",", "?", " ", "\n", ":"]:
                answer = answer[:-1].strip()
            # corner case 2: target = (B), prediction = B.
            if answer and answer[0] == "(" and answer[-1] == ")":
                answer = answer[1:-1].strip()
            extracted_letters.append(answer)

    return extracted_letters

def get_device_and_set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device

def main(config):
    devices = get_device_and_set_seed(config.seed)

    config.device = devices

    config.resume = config.resume_checkpoint not in [None, "None"]

    print('-' * 71)
    print(f"device: {config.device}")
    print(f"resume_check: {config.resume_checkpoint}")
    print(f"mode: {config.mode}")
    print('-' * 71)

    if config.mode == 'train':
        train(config)
    elif config.mode == 'test':
        test(config)
    else:
        raise ValueError('Invalid mode')

if __name__ == '__main__':
    # model_config = ModelConfig()
    # train(model_config)
    __spec__ = None
    parser = argparse.ArgumentParser(description="LLMGNP")

    ### Dataset
    parser.add_argument('--dataset', default='pubmedqa', help='dataset name')
    parser.add_argument('--data_dir', default='data', type=str, help='Path to the data directory')
    parser.add_argument('--ent_emb_paths', default='umls/ent_embed_coder.npy', help='sources for entity embeddings')
    parser.add_argument('--pad_node', default=-1, type=int)
    #propmt
    parser.add_argument('--context_node', default=False, type=utils.bool_flag, nargs='?', const=True)
    parser.add_argument('--context_pad', default=-2, type=int)
    parser.add_argument('--add_edge_attr', default='no', choices=['no', 'solo_label', 'semantic'])
    parser.add_argument('--use_char_options_format', default=False, type=utils.bool_flag, nargs='?', const=True)
    parser.add_argument('--prompt_context', default=True, type=utils.bool_flag, nargs='?', const=True)
    parser.add_argument('--prompt_Lanswer', default=False, type=utils.bool_flag, nargs='?', const=True)
    #subgraph
    parser.add_argument('--num_subgraphs', default=1, type=int)
    parser.add_argument('--num_nodes', default=200, type=int)
    parser.add_argument('--sub_graphs_choice', default='all', choices=['all', 'solo_question', 'solo_option'])
    ###Tune-P
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--epochs', default=50, type=int, help='total number of training epochs to perform.')
    parser.add_argument('--batch_size', default=8, type=int)
    #Opt
    parser.add_argument('--optim', default='adamw', type=str)
    parser.add_argument('--lr_schedule', default='fixed', choices=['fixed', 'warmup_linear', 'warmup_constant'], help='learning rate scheduler')
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='max grad norm (0 to disable)')
    parser.add_argument('--weight_decay', default=1e-2, type=float, help='l2 weight decay strength')
    
    #Task_weight
    parser.add_argument('--llm_task', type=float, default=1.0, help='Task weight for the LLM')
    parser.add_argument('--lp_task', type=float, default=0.1, help='Task weight for the LinkPred task')

    #-------Model-------
    ###LLM Model
    parser.add_argument('--model_name',  default='google/flan-t5-small', help='encoder type')
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--max_tag_len', default=128, type=int)
    parser.add_argument('--llm_frozen', default=True, type=utils.bool_flag, nargs='?', const=True)
    #LLM decoder modity
    parser.add_argument('--xattn_heads', default=8, type=int)
    parser.add_argument('--xattn_after', default=True, type=utils.bool_flag, nargs='?', const=True)
    ###GNP Model
    parser.add_argument('--add_gnp', default=False, type=utils.bool_flag, nargs='?', const=True)
    parser.add_argument('--cross_gnp', default=False, type=utils.bool_flag, nargs='?', const=True)
    parser.add_argument('--cross_gnp_num', default=5, type=int, help='attn_heads of the GNN layers')
    parser.add_argument('--cross_gnp_choice', default=1, type=int, help='attn_heads of the GNN layers')
    
    #initial_embed
    parser.add_argument('--random_ent_emb', default=False, type=utils.bool_flag, nargs='?', const=True, help='Whether to use randomly initialized learnable entity embeddings or not.')
    parser.add_argument('--freeze_ent_emb', default=True, type=utils.bool_flag, nargs='?', const=True, help='Whether to freeze the entity embedding layer.')
    #FFN
    parser.add_argument('--ffn_mult', default=4, type=int)
    #Gnn
    parser.add_argument('--gnn_layers', default=4, type=int, help='numbers of the GNN layers')
    parser.add_argument('--gnn_dim', default=1024, type=int, help='dimension of the GNN layers')
    parser.add_argument('--gnn_heads', default=2, type=int, help='attn_heads of the GNN layers')
    parser.add_argument('--gnn_norm', default=False, type=utils.bool_flag, nargs='?', const=True, help='encoder type')
    parser.add_argument('--gnn_residual', default='no', choices=['no', 'simple', 'linear'])
    #Self-Attention
    parser.add_argument('--self_attn_layers', default=1, type=int, help='numbers of the self-attention layers')
    parser.add_argument('--self_attn_heads', default=2, type=int, help='attn_heads of the self-attention layers')
    #Cross-Attention
    parser.add_argument('--cross_attn_layers', default=1, type=int, help='numbers of the CMP layers')
    parser.add_argument('--cross_attn_heads', default=2, type=int, help='attn_heads of the GNN layers')
    
    ###Link Prediction
    parser.add_argument('--is_lp', default=True, type=utils.bool_flag, nargs='?', const=True)
    parser.add_argument('--link_drop_probability', type=float, default=0.1, help='To specify #target positive triples for LinkPred')
    parser.add_argument('--link_negative_sample_size', type=int, default=64, help='')
    parser.add_argument('--link_negative_adversarial_sampling', type=utils.bool_flag, default=True, help='')
    parser.add_argument('--link_negative_adversarial_sampling_temperature', type=float, default=1, help='')
    parser.add_argument('--link_regularizer_weight', type=float, default=0.01, help='')
    parser.add_argument('--scaled_distmult', type=utils.bool_flag, default=False, help='')
    #-------Model end-------

    #Activation
    parser.add_argument('--activation', default='gelu', type=str)
    parser.add_argument('--pre_norm', default=True, type=utils.bool_flag, nargs='?', const=True)
    parser.add_argument('--init_range', default=0.02, type=float, help='stddev when initializing with normal distribution')
    parser.add_argument('--dropout_emb', type=float, default=0.2, help='dropout for GNN layers')
    parser.add_argument('--dropout_ffn', type=float, default=0.2, help='dropout for GNN layers')
    parser.add_argument('--dropout_gnn', type=float, default=0.2, help='dropout for GNN layers')
    parser.add_argument('--dropout_self_attn', type=float, default=0.1, help='dropout for GNN layers')
    parser.add_argument('--dropout_cross_attn', type=float, default=0.1, help='dropout for GNN layers')

    #save and log
    parser.add_argument('--save_dir', default=f'./saved_models/', help='model output directory')
    parser.add_argument('--save_model', default=1, type=float, help="0: do not save model checkpoints. 1: save if best dev. 2: save always")
    parser.add_argument('--load_model_path', default=None, help="The model checkpoint to load in the evaluation mode.")

    #Test-Debug use
    parser.add_argument('--log_interval', default=10, type=int)
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument("--resume_checkpoint", default=None, type=str,
                        help="The checkpoint to resume training from.")
    parser.add_argument('--mode', default='train', choices=['train', 'test'], help='run training or evaluation')
    
    args = parser.parse_args()
    config = ModelConfig(vars(args))

    main(config)



