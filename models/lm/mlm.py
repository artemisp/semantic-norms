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
    with open(f"data/datasets/{DATASET}/queries" + prompt_type + ".prop", "r") as f:
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