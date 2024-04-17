from transformers import AutoTokenizer, AutoModel
from torch import cuda
import numpy as np
import pandas as pd
import torch
import h5py
import os
import json

device = 'cuda:0' if cuda.is_available() else 'cpu'
coder_tokenizer = AutoTokenizer.from_pretrained("GanjinZero/coder_eng_pp")
coder_model = AutoModel.from_pretrained("GanjinZero/coder_eng_pp").to(device)
coder_model.output_hidden_states = False

batch_size = 128

# Best CODER results are with [CLS] representations and normalization (default)
def get_bert_embed(phrase_list, model, tokenizer, normalize=True, summary_method="CLS"):
    # TOKENIZATION
    input_ids = []
    for phrase in phrase_list:
        # (1) Tokenize the sentence.
        # (2) Prepend the `[CLS]` token to the start.
        # (3) Append the `[SEP]` token to the end.
        # (4) Map tokens to their IDs.
        # (5) Pad or truncate the sentence to `max_length`
        # (6) Create attention masks for [PAD] tokens.
        input_ids.append(tokenizer(
            phrase,
            max_length=32, # UMLS terms are short
            add_special_tokens=True,
            truncation=True,
            pad_to_max_length=True)['input_ids'])

    # INFERENCE MODE ON
    model.eval()

    # COMPUTE EMBEDDINGS ACCORDING TO THE SPECIFIED BATCH-SIZE
    # (e.g., max_length=32, batch_size=64 --> 2 phrase embeddings at a time)
    count = len(input_ids) # n total tokens
    now_count = 0
    with torch.no_grad():
        while now_count < count:
            batch_input_gpu = torch.LongTensor(input_ids[
                now_count:min(now_count + batch_size, count)]).to(device)
            if summary_method == "CLS":
                embed = model(batch_input_gpu)[1]
            if summary_method == "MEAN":
                embed = torch.mean(model(batch_input_gpu)[0], dim=1)
            if normalize:
                embed_norm = torch.norm(
                    embed, p=2, dim=1, keepdim=True).clamp(min=1e-12)
                embed = embed / embed_norm
            # Move embedding on CPU and convert it to a numpy array
            embed_np = embed.cpu().detach().numpy()
            # Update indeces for batch processing
            if now_count == 0:
                output = embed_np
            else:
                output = np.concatenate((output, embed_np), axis=0)
            now_count = min(now_count + batch_size, count)
    return output

if __name__ == '__main__':

    triplets = pd.read_csv('./data/umls_triplet_name.csv')
    head = triplets['head'].tolist()
    tail = triplets['tail'].tolist()
    entities = list(set(head + tail))

    #get corresponding cui
    with open(os.getcwd() + "/data/concept_name.json", "r") as json_file:
        concept_name = json.load(json_file)
    name_concept = {v:k for k,v in concept_name.items()}

    cuis = []
    for entity in entities:
        cuis.append(name_concept[entity])

    #get embed
    entities_feat = get_bert_embed(entities, coder_model, coder_tokenizer)


    embed_file = os.getcwd() + '/data/coder_embed_entity.h5'
    #write
    with h5py.File(embed_file, 'w') as hf:
        hf.create_dataset('cui', data=cuis)
        hf.create_dataset('entity', data=entities)
        hf.create_dataset('embedding', data=np.array(entities_feat, dtype=np.float64))
