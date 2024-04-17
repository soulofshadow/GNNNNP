import os
import sys
sys.path.insert(0, sys.path[0]+"/../")

from tqdm import tqdm
import pandas as pd

from utils.load_triplet import Triplet
from predict_by_TEKGEN import get_predict

'''
1. triplets
considering the efficiency and all previous work i searched on internet, \\
UMLS's Metamap is better for retrieving CUI, but for training a LM, it is\\
better using some text sentences from Pubmed and Metamap to find the corresponding \\
CUI, then align it with MRREL.RRF to get the triplet that we want.  
'''
def get_triplets_memory(path, save_file=False):

    if save_file:
        data = Triplet(path)
        triplets = data.get_triplets_as_list()

        sentences = []
        for entity in tqdm(data.ents):
            serialized = [x for x in triplets if x[0] == entity]
            if serialized is None:
                continue
            sentence = ""
            temp = 0
            for triplet in serialized:
                if temp == 0:
                    sentence = ' '.join(triplet)
                    temp = 1
                else:
                    sentence += ', '
                    sentence += ' '.join(triplet[1:])
            sentences.append([entity, sentence])

        df = pd.DataFrame(columns=['entity', 'serialized_sentence'], data=sentences)
        df.to_csv('data/triplets_memory.csv',index_label=False)
        return df
    else:
        df = pd.read_csv('data/triplets_memory.csv')
        return df

'''
2. using TEKGEN generated corpus from subgraph

'''
def get_corpus_memory(path, save_file=False):
    #get the pre-finetuned Tekgen model
    #if not, train that first by using TEKGEN
    if save_file:
        dataset = pd.read_csv(path)
        df = get_predict(dataset)
        df.to_csv('data/kelm_memory.csv',index_label=False)
        return df
    else:
        df = pd.read_csv('data/kelm_memory.csv')
        return df
    
'''
3. cluster of entity (using CODER++)

'''


if __name__ == '__main__':

    #
    triplet_path = 'data/triplets.txt'
    triplet_memory = get_triplets_memory(triplet_path, save_file=True)

    #
    corpus_path = 'data/subgraph_dataset.csv'
    corpus_memory = get_corpus_memory(corpus_path, save_file=True)

    #
    


    
    

    

