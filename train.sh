#!/bin/bash
# python file to exec

#PARAMS:
#1: dataset
#2: dataset's num_subgraph sets
#3: LLM mode
#4: LLM name
#5: LLM freeze
#6: GNP module
#7: Checkpoint


#SLURM use
# ********************************
# dataset=${1}
# sub_graphs_choice=${2}

# mode=${3}
# model_name=${4}
# llm_frozen=${5}

# add_gnp=${6}
# cross_gnp=${7}
# load_model_path=${8}
# ********************************

#debug use
# ********************************
dataset="pubmedqa"
sub_graphs_choice='all'

mode="test"
model_name="google/flan-t5-base"
llm_frozen=true

add_gnp=false
cross_gnp=false
load_model_path=None
# ********************************


#model detail mannually modify params
#######if cross_gnp=True
#decoder of T5
cross_gnp_num=5
cross_gnp_choice=1
xattn_heads=8
xattn_after=false

#######GNP block
#GAT
gnn_layers=4
gnn_dim=1024
gnn_heads=2
gnn_norm=false
gnn_residual='linear'
#ATTN
pre_norm=true
self_attn_layers=1
self_attn_heads=2
cross_attn_layers=1
cross_attn_heads=2
#Link Prediction
is_lp=true

#######data preprocess
context_node=false
add_edge_attr='no'
ent_emb_paths='umls/ent_emb_blbertL.npy'  #the initial node embedding 1024dim
use_char_options_format=false  #model predict options or answer strings
prompt_context=true 
prompt_Lanswer=false
num_nodes=200  # max_nodes length
max_seq_len=512
max_tag_len=128

#######hyP
learning_rate=1e-4
epochs=50
batch_size=8
optim='radam'
lr_schedule='warmup_linear'
warmup_ratio=0.1
llm_task=1.0
lp_task=0.1


# Run the script
python3 train.py \
    --dataset $dataset \
    --sub_graphs_choice $sub_graphs_choice \
    --mode $mode --model_name $model_name --llm_frozen $llm_frozen \
    --add_gnp $add_gnp \
    --cross_gnp $cross_gnp \
    --load_model_path $load_model_path \
    --cross_gnp_num $cross_gnp_num --cross_gnp_choice $cross_gnp_choice \
    --xattn_heads $xattn_heads --xattn_after $xattn_after \
    --gnn_layers $gnn_layers --gnn_dim $gnn_dim --gnn_heads $gnn_heads \
    --gnn_residual $gnn_residual --gnn_norm $gnn_norm \
    --pre_norm $pre_norm \
    --self_attn_layers $self_attn_layers --self_attn_heads $self_attn_heads \
    --cross_attn_layers $cross_attn_layers --cross_attn_heads $cross_attn_heads \
    --is_lp $is_lp \
    --context_node $context_node \
    --add_edge_attr $add_edge_attr \
    --ent_emb_paths $ent_emb_paths \
    --use_char_options_format $use_char_options_format \
    --prompt_context $prompt_context --prompt_Lanswer $prompt_Lanswer \
    --num_nodes $num_nodes \
    --max_seq_len $max_seq_len --max_tag_len $max_tag_len \
    --learning_rate $learning_rate --epochs $epochs --batch_size $batch_size \
    --optim $optim --lr_schedule $lr_schedule --warmup_ratio $warmup_ratio \
    --llm_task $llm_task --lp_task $lp_task \