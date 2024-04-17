import argparse
import networkx as nx
import nltk
import json
import csv
import numpy as np
from tqdm import tqdm
import os

from load_umls import UMLS

def generate_umls_data(meta):

    print(f"---Starting read from {meta} Meta dataset...---")

    umls = UMLS(f'./public_mm/META_data/META_{meta}')

    #build concepts 
    concepts_file = "./data/umls/concepts.txt"
    with open(concepts_file, 'w') as file:
        for item in umls.cui:
            file.write("%s\n" % item)

    #build concepts to name
    # cui-name
    cui2name = {}
    for key, value in tqdm(umls.str2cui.items()):
        if value not in cui2name.keys():
            cui2name[value] = key.lower()

    for key, value in tqdm(umls.cui2str.items()):
        if key not in cui2name.keys():
            cui2name[key] = value.pop().lower()

    concepts2name_file = "./data/umls/concept_names.txt"
    with open(concepts2name_file, 'w') as file:
        for key, value in cui2name.items():
            file.write(f'{key}\t{value}\n')

    
    #build relations
    umls_file = "./data/umls/umls.csv"
    relation_file = "./data/umls/relations.txt"

    csvf_edges = open(umls_file, "w", newline='', encoding='utf-8')
    w_entity = csv.writer(csvf_edges)
    rels = []
    for t in tqdm(umls.rel):
        triplet = t.strip().split("\t")
        if len(triplet) != 4:
            continue;
        head = triplet[0]
        tail = triplet[1]
        rel = triplet[3]
        if rel not in rels:
            rels.append(rel)
        w_entity.writerow((head, tail, rel))
    csvf_edges.close()


    with open(relation_file, 'w') as file:
        for item in rels:
            file.write("%s\n" % item)
    print(f"---UMLS dataset {meta} generated successfully---")

def construct_umls_graph(meta, prune=True):
    if meta == 'FULL':
        print('---FULL situation plz use Neo4j...---')
        return

    print('---generating UMLS graph file...---')
    blacklist = set()

    #load entities
    concepts_file = "./data/umls/concepts.txt"
    id2concept = [w.strip() for w in open(concepts_file)]
    concept2id = {w: i for i, w in enumerate(id2concept)}

    #load relations
    relation_file = "./data/umls/relations.txt"
    id2relation = [r.strip() for r in open(relation_file)]
    relation2id = {r: i for i, r in enumerate(id2relation)}

    #load triplets
    umls_file = "./data/umls/umls.csv"
    graph = nx.MultiDiGraph()
    nrow = sum(1 for _ in open(umls_file, 'r', encoding='utf-8'))
    with open(umls_file, "r", encoding="utf8") as fin:

        def not_save(cpt):
            if cpt in blacklist:
                return True
            return False
        
        attrs = set()

        for line in tqdm(fin, total=nrow):
            ls = line.strip().split(',')
            subj = concept2id[ls[0]]
            obj = concept2id[ls[1]]
            rel = relation2id[ls[2]]
            weight = float(1)

            if prune and (not_save(ls[1]) or not_save(ls[2]) or id2relation[rel] == "hascontext"):
                continue
            if subj == obj:  # delete loops
                continue
            if (subj, obj, rel) not in attrs:
                graph.add_edge(subj, obj, rel=rel, weight=weight)
                attrs.add((subj, obj, rel))
                graph.add_edge(obj, subj, rel=rel + len(relation2id), weight=weight)
                attrs.add((obj, subj, rel + len(relation2id)))

    output_path = "./data/umls/umls.graph"
    nx.write_gpickle(graph, output_path)
    print(f"graph file saved to {output_path}")
    print()

def main(args):
    
    files = os.listdir('data')
    if 'umls' not in files:
        os.system(f'mkdir -p data/umls')
        generate_umls_data(args.UMLS)
        construct_umls_graph(args.UMLS)
    else:
        umls_data = ['concept_name.json', 'concepts.json', 'relations.json', 'umls.csv', 'umls.graph']
        files = os.listdir('data/umls')
        for u in umls_data[:-1]:
            if u not in files:
                generate_umls_data(args.UMLS)
                break
        if umls_data[-1] not in files:
            construct_umls_graph(args.UMLS)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--UMLS', default='FULL', 
                        help='choose which dataset to use whether FULL or subset of Disease')
    args = parser.parse_args()

    print(f'Used {args.UMLS} meta data to construct graph')

    main(args)

