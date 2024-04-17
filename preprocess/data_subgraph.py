
import json
import networkx as nx
import pickle
from tqdm import tqdm
import itertools
import random
import os

# from preprocess.neo4j import Neo4jConnection

concept2id = None
concept2name = None
id2concept = None
relation2id = None
id2relation = None

NET = None
#for subgraph extra nodes

def load_resources():
    global concept2id, id2concept, relation2id, id2relation, concept2name

    #load concepts
    concepts_file = "./data/umls/concepts.txt"
    id2concept = [w.strip() for w in open(concepts_file)]
    concept2id = {w: i for i, w in enumerate(id2concept)}

    #load concept to name
    concept2name_file = "./data/umls/concept_names.txt"
    concept2name = {}
    for line in open(concept2name_file):
        c, n = line.strip().split('\t')
        concept2name[c] = n
    
    #load relation
    relation_file = "./data/umls/relations.txt"
    id2relation = [r.strip() for r in open(relation_file)]
    relation2id = {r: i for i, r in enumerate(id2relation)}

def load_cpnet():
    global NET
    graph_file = './data/umls/umls.graph'

    # NET = nx.read_gpickle(graph_file)
    with open(graph_file, 'rb') as f:
        NET = pickle.load(f)


def retrieval_subgraph(qc):

    #transfer to id and prune by graph concept
    qc_id = [concept2id[c] for c in qc if c in id2concept]

    origin_nodes = set(qc_id)
    extra_nodes = set()

    neibors = {}
    for n in origin_nodes:
        if n in NET.nodes():
            neibors[n] = set(NET.neighbors(n))
        else:
            neibors[n] = set()
    for a, b in itertools.combinations(origin_nodes, 2):
        extra_nodes |= set(neibors[a]) & set(neibors[b]) 
    extra_nodes = extra_nodes - origin_nodes

    '''
    output should contain nodes, edges and edge_type
    '''
    nodes_list = sorted(origin_nodes) + sorted(extra_nodes)
    #find edges
    edges_list = []
    edge_types_list = []
    for a, b in itertools.combinations(nodes_list, 2):
        if NET.has_edge(a, b):
            edges_list.append((a, b)) 
            edge_type = NET.get_edge_data(a, b)[0]['rel']
            edge_types_list.append(edge_type)
    res = {'nodes':nodes_list, 'edges': edges_list, 'edge_types':edge_types_list}
    return res

def retrieval_option_subgraph(qc, ac):

    qc_id = [concept2id[c] for c in qc if c in id2concept]
    ac_id = [concept2id[c] for c in ac if c in id2concept]

    if not ac_id:
        return {'nodes':[], 'edges': [], 'edge_types':[]}

    qc_neigbor = {}
    for n in qc_id:
        if n in NET.nodes():
            qc_neigbor[n] = set(NET.neighbors(n))
        else:
            qc_neigbor[n] = set()

    ac_neigbor = {}
    for n in ac_id:
        if n in NET.nodes():
            ac_neigbor[n] = set(NET.neighbors(n))
        else:
            ac_neigbor[n] = set()

    linked_qc_nodes = set()
    extra_nodes = set()
    for ac_node in ac_id:
        for qc_node in qc_id:
            hop2_nodes = set(ac_neigbor[ac_node]) & set(qc_neigbor[qc_node]) 
            if hop2_nodes:
                linked_qc_nodes.add(qc_node)
                extra_nodes |= hop2_nodes

    if not linked_qc_nodes:
        return {'nodes':[], 'edges': [], 'edge_types':[]}

    all_graph_nodes = list(set(ac_id).union(extra_nodes, linked_qc_nodes))
    edges_list = []
    edge_types_list = []
    for a, b in itertools.combinations(all_graph_nodes, 2):
        if NET.has_edge(a, b):
            edges_list.append((a, b)) 
            edge_type = NET.get_edge_data(a, b)[0]['rel']
            edge_types_list.append(edge_type)
    res = {'nodes':all_graph_nodes, 'edges': edges_list, 'edge_types':edge_types_list}
    return res

def retrieval_all_option_subgraph(qc, ac):

    qc_id = [concept2id[c] for c in qc if c in id2concept]
    if ac is not None:
        ac_id = [concept2id[c] for c in ac if c in id2concept]
    else:
        ac_id = []

    qc_neigbor = {}
    for n in qc_id:
        if n in NET.nodes():
            qc_neigbor[n] = set(NET.neighbors(n))
        else:
            qc_neigbor[n] = set()

    ac_neigbor = {}
    for n in ac_id:
        if n in NET.nodes():
            ac_neigbor[n] = set(NET.neighbors(n))
        else:
            ac_neigbor[n] = set()

    linked_qc_nodes = set()
    extra_nodes = set()
    for ac_node in ac_id:
        for qc_node in qc_id:
            hop2_nodes = set(ac_neigbor[ac_node]) & set(qc_neigbor[qc_node]) 
            if hop2_nodes:
                linked_qc_nodes.add(qc_node)
                extra_nodes |= hop2_nodes

    all_graph_nodes = list(set(ac_id).union(extra_nodes, linked_qc_nodes))

    origin_nodes = set(ac_id) | set(qc_id)
    neibors = {}
    for n in origin_nodes:
        if n in NET.nodes():
            neibors[n] = set(NET.neighbors(n))
        else:
            neibors[n] = set()
    for a, b in itertools.combinations(origin_nodes, 2):
        extra_nodes |= set(neibors[a]) & set(neibors[b]) 
    extra_nodes = extra_nodes - origin_nodes
    extra_nodes = list(extra_nodes - set(all_graph_nodes))

    all_nodes = all_graph_nodes + extra_nodes
    unique_list = list(dict.fromkeys(all_nodes))

    #find edges
    edges_list = []
    edge_types_list = []
    for a, b in itertools.combinations(unique_list, 2):
        if NET.has_edge(a, b):
            edges_list.append((a, b)) 
            edge_type = NET.get_edge_data(a, b)[0]['rel']
            edge_types_list.append(edge_type)
    res = {'nodes':unique_list, 'edges': edges_list, 'edge_types':edge_types_list}
    return res

def match_subgraph(questions, options):
    assert len(questions) == len(options), (len(questions), len(options))
    res = []

    for i in tqdm(range(len(questions))):
        option = options[i]
        #solo q
        if not option:
            res.append([retrieval_subgraph(questions[i])])
        
        #each option
        if option:
            # r = []
            # choices = len(option)
            # for j in range(choices):
            #     #question graph
            #     if j == 0:
            #         r.append(retrieval_subgraph(questions[i]))
            #     #option graph
            #     else:
            #         r.append(retrieval_option_subgraph(questions[i], option[j]))
            # res.append(r)

            #
            flat_list = [item for sublist in option if sublist is not None for item in sublist]
            res.append([retrieval_all_option_subgraph(questions[i], flat_list)])

    return res
        
def build_subgraph(file_path, file_type, test_subgraph=False):

    global concept2id, id2concept, relation2id, id2relation, concept2name
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_resources()    

    global NET
    if NET is None:
        load_cpnet()
        # NET = Neo4jConnection(uri="neo4j://localhost:7687", user="neo4j", pwd="12345678")

    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)

    if test_subgraph:
        data = data[:3]

    if file_type == 'medqa':
        concept = [d['concept'] for d in data]
        option_graph = [None] * len(concept)
        option0 = [d['option0'] for d in data]
        option1 = [d['option1'] for d in data]
        option2 = [d['option2'] for d in data]
        option3 = [d['option3'] for d in data]
        options = [list(pair) for pair in zip(option_graph, option0, option1, option2, option3)]
        res = match_subgraph(concept, options)
    elif file_type == 'pubmedqa':
        concept = [d['concept'] for d in data]
        options = [None] * len(concept)
        res = match_subgraph(concept, options)
    elif file_type == 'bioasq':
        concept = [d['concept'] for d in data]
        options = [None] * len(concept)
        res = match_subgraph(concept, options)



    # check_path(output_path)
    output_type = 'subgraphed'
    file_dir = f"{os.sep}".join(file_path.split(os.sep)[:-2])  #father dir
    file_name = "".join(file_path.split(os.sep)[-1].split('.')[:-1])
    output_path = os.path.join(file_dir, output_type, file_name + '.jsonl')
    os.system('mkdir -p {}'.format(os.path.dirname(output_path)))
    with open(output_path, 'w') as fout:
        for dic in res:
            fout.write(json.dumps(dic) + '\n')

    print(f'grounded concepts saved to {output_path}')
    print()


if __name__ == '__main__':

    pass