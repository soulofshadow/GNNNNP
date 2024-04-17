#!/bin/bash

#PARAMS:
#1: dataset
#2: dataset's num_subgraph sets
#3: LLM mode
#4: LLM name
#5: LLM freeze
#6: GNP module
#7: CrossGNN module
#8:Checkpoint

#1
dataset_pubmed="pubmedqa"
dataset_bioasq="bioasq"
dataset_medqa="medqa"
#2
sub_graphs_choice_all="all"
sub_graphs_choice_question="solo_question"
sub_graphs_choice_options="solo_option"
#3
mode_train="train"
mode_test="test"
#4
model_name="google/flan-t5-base"
#5
llm_frozen=true
llm_unfrozen=false
#6
gnp_turnon=true
gnp_turnoff=false
#7
cross_turnon=true
cross_turoff=false
#8
load_model_path=None

#check for permission
script="run_docker.sh"
if [ ! -x "$script" ]; then
    echo "Adding execute permission to $script"
    chmod +x "$script"
fi


####Pubmed
#pubmedqa GNP
# sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 -w faretra run_docker.sh \
#     "$dataset_pubmed" \
#     "$sub_graphs_choice_all" \
#     "$mode_train" \
#     "$model_name" \
#     "$llm_frozen" \
#     "$gnp_turnon" \
#     "$cross_turoff" \
#     "$load_model_path"

# #pubmedqa CrossGNN
# sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 -w faretra run_docker.sh \
#     "$dataset_pubmed" \
#     "$sub_graphs_choice_all" \
#     "$mode_train" \
#     "$model_name" \
#     "$llm_frozen" \
#     "$gnp_turnoff" \
#     "$cross_turnon" \
#     "$load_model_path"

# #pubmedqa GNP + CrossGNN
# sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 -w faretra run_docker.sh \
#     "$dataset_pubmed" \
#     "$sub_graphs_choice_all" \
#     "$mode_train" \
#     "$model_name" \
#     "$llm_frozen" \
#     "$gnp_turnon" \
#     "$cross_turnon" \
#     "$load_model_path"

    

####MedQA
#GNP
sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 -w faretra run_docker.sh \
    "$dataset_medqa" \
    "$sub_graphs_choice_all" \
    "$mode_train" \
    "$model_name" \
    "$llm_frozen" \
    "$gnp_turnon" \
    "$cross_turoff" \
    "$load_model_path"

sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 -w faretra run_docker.sh \
    "$dataset_medqa" \
    "$sub_graphs_choice_question" \
    "$mode_train" \
    "$model_name" \
    "$llm_frozen" \
    "$gnp_turnon" \
    "$cross_turoff" \
    "$load_model_path"

sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 -w faretra run_docker.sh \
    "$dataset_medqa" \
    "$sub_graphs_choice_options" \
    "$mode_train" \
    "$model_name" \
    "$llm_frozen" \
    "$gnp_turnon" \
    "$cross_turoff" \
    "$load_model_path"


#CrossGNN
sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 -w faretra run_docker.sh \
    "$dataset_medqa" \
    "$sub_graphs_choice_all" \
    "$mode_train" \
    "$model_name" \
    "$llm_frozen" \
    "$gnp_turnoff" \
    "$cross_turnon" \
    "$load_model_path"

#GNP + CrossGNN
sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 -w faretra run_docker.sh \
    "$dataset_medqa" \
    "$sub_graphs_choice_all" \
    "$mode_train" \
    "$model_name" \
    "$llm_frozen" \
    "$gnp_turnon" \
    "$cross_turnon" \
    "$load_model_path"

# #BIOasq
# sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 -w faretra run_docker.sh \
#     "$dataset_bioasq" \
#     "$sub_graphs_choice_all" \
#     "$mode_train" \
#     "$model_name" \
#     "$llm_frozen" \
#     "$gnp_turnon" \
#     "$cross_turoff" \
#     "$load_model_path"

# #pubmedqa CrossGNN
# sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 -w faretra run_docker.sh \
#     "$dataset_bioasq" \
#     "$sub_graphs_choice_all" \
#     "$mode_train" \
#     "$model_name" \
#     "$llm_frozen" \
#     "$gnp_turnoff" \
#     "$cross_turnon" \
#     "$load_model_path"

# #pubmedqa + CrossGNN + fine tune
sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 -w faretra run_docker.sh \
    "$dataset_bioasq" \
    "$sub_graphs_choice_all" \
    "$mode_train" \
    "$model_name" \
    "$llm_unfrozen" \
    "$gnp_turnoff" \
    "$cross_turnon" \
    "$load_model_path"







