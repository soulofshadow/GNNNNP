import json
import argparse
from multiprocessing import Pool
import multiprocessing
import os
from tqdm import tqdm
import numpy as np

from preprocess.mmlrestclient import METAMAPOnline
from preprocess.mmlocalclient import METAMAP

import spacy
import scispacy
from scispacy.linking import EntityLinker

#global
MAP_ = None  # 1:using NLP_
NLP_ = None

def load_entity_linker(threshold=0.90):
    nlp = spacy.load("en_core_sci_sm")
    nlp.add_pipe("scispacy_linker", 
                 config={"resolve_abbreviations": True, "linker_name": "umls", "threshold": threshold})
    return nlp

def get_entities_by_spacy(sent):
    doc = NLP_(sent)
    mentions = doc.ents

    mentioned_concepts = set()
    for mention in mentions:
        concepts = mention._.kb_ents
        for concept in concepts:
            mentioned_concepts.add(concept[0])
    return mentioned_concepts

def get_entities(sent):
    if MAP_ == 1:
        return get_entities_by_spacy(sent)
    else:
        return MAP_.get_entities(sent)

def ground_qa_pair(question, options):

    question_concepts = get_entities(question)
    all_option_concept = set()

    if options:
        choices = len(options)
        re = {}
        for i in range(choices):
            locals()[f'option{i}'] = get_entities(options[i]) #set
            re[f'option{i}'] = list(locals()[f'option{i}']) #list
            all_option_concept.update(locals()[f'option{i}'])
            
        question_concepts = list(question_concepts - all_option_concept)
        dic = {'concept': question_concepts}
        dic.update(re)
        return dic
    else:
        dic = {'concept': list(question_concepts)}
        return dic

def match_mentioned_concepts(questions, options):
    assert len(questions) == len(options), (len(questions), len(options))
    
    res = []
    for i in tqdm(range(len(questions))):
        res.append(ground_qa_pair(questions[i], options[i]))

    return res

def map_concept_ner(map_way, file_path, file_type, test_metamap=False):
    global MAP_, NLP_
    if map_way == 'spacy' and NLP_ is None:
        NLP_ = load_entity_linker()
        MAP_ = 1
    else:
        if MAP_ is None:
            if map_way == 'local':
                MAP_ = METAMAP()
            elif map_way == 'online' :
                MAP_ = METAMAPOnline()

    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)

    #test metamap
    if test_metamap:
        data = data[:3]

    # for diff dataset 
    if file_type == 'medqa':
        questions = [d['question'] for d in data]
        option0 = [d['options']['A'] for d in data]
        option1 = [d['options']['B'] for d in data]
        option2 = [d['options']['C'] for d in data]
        option3 = [d['options']['D'] for d in data]
        options = [list(pair) for pair in zip(option0, option1, option2, option3)]
        res = match_mentioned_concepts(questions, options)
    elif file_type == 'pubmedqa':
        questions = [d['question'] for d in data]
        contexts = [d['context'] for d in data]
        longanswers = [d['longanswer'] for d in data]
        allinputs = [" ".join(elements) for elements in zip(questions, contexts, longanswers)]
        options = [None] * len(allinputs)
        res = match_mentioned_concepts(allinputs, options)
    elif file_type == 'bioasq':
        questions = [d['question'] for d in data]
        contexts = [d['context'] for d in data]
        allinputs = [" ".join(elements) for elements in zip(questions, contexts)]
        options = [None] * len(allinputs)
        res = match_mentioned_concepts(allinputs, options)

    # check_path(output_path)
    output_type = 'mapped'
    file_dir = f"{os.sep}".join(file_path.split(os.sep)[:-2]) #father dir
    file_name = "".join(file_path.split(os.sep)[-1].split('.')[:-1])
    output_path = os.path.join(file_dir, output_type, file_name + '.jsonl')
    os.system('mkdir -p {}'.format(os.path.dirname(output_path)))
    with open(output_path, 'w') as fout:
        for dic in res:
            fout.write(json.dumps(dic) + '\n')

    print(f'grounded concepts saved to {output_path}')
    print()

    return 

if __name__ == '__main__':

    pass