#path import and add root project path to sys
import os
import sys
sys.path.insert(0, sys.path[0]+"/../")

#normal import for data processing
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import time

#third-party package
from umlsrat.api.metathesaurus import MetaThesaurus

#utils
from utils.load_umls import UMLS

#build the triplets from umls
def get_triplet(umls):
    
    umls_triplet = []
    for rel in umls.rel:
        triplet = rel.strip().split("\t")
        if len(triplet) != 4:
            continue;

        head = triplet[0]
        tail= triplet[1]
        relation = triplet[3]

        tri = [head, relation, tail]
        umls_triplet.append(tri)

    print("triplets count:", len(umls_triplet))

    umls_triplet = pd.DataFrame(umls_triplet)
    umls_triplet.columns = ['head', 'relation', 'tail']
    umls_triplet.drop_duplicates()
    umls_triplet.to_csv('./data/umls_triplet.csv', index=False)

# os.environ['UMLS_API_KEY'] = '68ceaca1-84b1-4a6c-babb-f749e6af9df7'
# def get_cui_name(api, cui):
#     # for example get information for a known concept
#     info = api.get_concept(cui)
#     if info is None:
#         return None
#     else:
#         return info.get("name")

if __name__ == '__main__':

    umls = UMLS('./data/')

    if not os.path.exists(os.getcwd() + '/data/umls_triplet.csv'):
        get_triplet(umls)

    '''
    for now the triplet is stored as CUI,CUI,REL.
    need to transfer it to the name of CUI by umls api
    '''
    if not os.path.exists(os.getcwd() + '/data/umls_triplet_name.csv'):
        cui2name = {}
        for key, value in umls.str2cui.items():
            if value not in cui2name.keys():
                cui2name[value] = key

        umls_triplets = pd.read_csv('./data/umls_triplet.csv')

        row_to_delete = []
        for index, item in tqdm(umls_triplets.iterrows()):
            if item['head'] not in cui2name.keys() or item['tail'] not in cui2name.keys():
                row_to_delete.append(index)
                continue
            
            item['head'] = cui2name[item['head']]
            item['tail'] = cui2name[item['tail']]

        triplets = umls_triplets.drop(row_to_delete)
        triplets.to_csv('./data/umls_triplet_name.csv', index=False)


    


        





    

    










