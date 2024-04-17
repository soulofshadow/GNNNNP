# zeng-Theis Project

**Folders**

* backups: other tried solutions
* data: including all QA datasets and UMLS (triplets + graph + entity embeddings)
* modeling: _gnn: design for graph network module; _gnp: design for graph prompt module; _llm: connect graph prompt and language model training and inference; _t5: modified version of t5 mdoel (added gated cross attention in decoder).
* preprocess: used for building paired QA samples and UMLS subgraphs
* utils: data loading and network designs


**Main File**

train.py: containing all process controled by argparse settings


**RUN instructions**

1. ***For slurm training:***

sbatch_script.sh -> run_docker.sh -> train.sh

    ##define which QA dataset you wanna test on 'sbatch_script.sh', run it, then it will call other scripts as above.

    ##if wanna change model parameters, change it on train.sh.

1. ***For local debug:***

train.sh

    ##Comment out '#SLURM use' this part of the code in file train.sh, and use '#debug use' this part of the code to run it locally.
