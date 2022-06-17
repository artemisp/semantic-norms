import sys
sys.path.insert(1, '../..')
import eval
import pickle
import numpy as np
from tqdm import tqdm
from nltk.corpus import wordnet as wn
import os
from transformers import pipeline

retrieval_based = True
add_def = True

DATASET = "concept_properties" # "feature_norms", "memory_colors"

def get_prompts(prompt_type):
    noun2sent = {}
    with open(f"../data/datasets/{DATASET}/queries" + prompt_type + ".prop", "r") as f:
        for raw_data in f.readlines():
            noun = raw_data.split(" :: ")[0]
            sent = raw_data.split(" :: ")[1][:-1]
            noun2sent[noun] = sent
    return noun2sent

def get_def(noun):
    try:
        synsets = wn.synsets(noun, pos=wn.NOUN)
        definition = synsets[0].definition()
    except:
        definition = "a common North American jay with a blue crest, back, wings, and tail"
    return definition

if __name__ == "__main__":
    noun2prop = pickle.load(open(f"../data/datasets/{DATASET}/noun2property/noun2prop.p", "rb"))
    model_name = 'bert-large-uncased'
    unmasker = pipeline('fill-mask', model=model_name, device = 0)

    candidate_adjs = []
    for noun, props in noun2prop.items():
        candidate_adjs += props
    candidate_adjs = list(set(candidate_adjs))
    
    noun2sent = get_prompts(prompt_type = "plural_most")
    noun2predicts = {}
    for noun, sent in tqdm(noun2sent.items()):
        noun2predicts[noun] = []
        if add_def:
            sent = noun + ' is ' + get_def(noun) + '. ' + sent

        if 'roberta' in model_name or 'bart' in model_name or 'mpnet' in model_name:
            sent = sent.replace("[MASK]", "<mask>")

        if retrieval_based:
            for output in unmasker(sent, top_k = 10):
                noun2predicts[noun].append(output['token_str'].replace(' ', ''))
        else:
            for output in unmasker(sent, targets = candidate_adjs, top_k = 10):
                noun2predicts[noun].append(output['token_str'].replace(' ', ''))

    save_file_name = ''
    if add_def:
        save_file_name = save_file_name + "w_def"
    if retrieval_based:
        save_file_name = save_file_name + "_retri"               

    eval.evaluate(noun2predicts, noun2prop, PRINT=True)
    eval.evaluate_acc(noun2predicts, noun2prop,  PRINT=True)