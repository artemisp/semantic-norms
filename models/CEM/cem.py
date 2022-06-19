import pickle
import sys
from tqdm import tqdm
import numpy as np
sys.path.append('../..')
import eval
from difflib import SequenceMatcher
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

import math
def s(x):
    return 1 / (1 + math.e**(-x))


class CEM():
    def __init__(self, dataset, model_concrete='clip', model_abstract='roberta', predicted_concreteness=False):
        pass

DATASET = "feature_norms" # "concept_properties", "feature_norms", "memory_colors"

noun2prop = pickle.load(open(f"../data/datasets/{DATASET}/noun2property/noun2prop.p", "rb"))
gpt3_scores = pickle.load(open(f"../data/datasets/{DATASET}/GPT3/gpt3_predicts.txt", "rb"))
roberta_scores = pickle.load(open(f"../output/output_{DATASET}/roberta-large+singular_generally.p", "rb"))
bert_scores = pickle.load(open(f"../output/output_{DATASET}/bert-large-uncased+plural_most.p", "rb"))
vilt_scores = pickle.load(open(f"../output/output_{DATASET}/vilt+plural+10.p", "rb"))
clip_scores = pickle.load(open(f"../data/datasets/{DATASET}/CLIP/clip_scores.p", "rb"))
combined_scores = pickle.load(open(f"../data/datasets/{DATASET}/CEM/combine_scores.p", "rb"))
ngram_scores = pickle.load(open("../data/datasets/feature_norms/ngram_scores.p", "rb"))
gpt_scores = pickle.load(open(f"../output/output_{DATASET}/gpt2-large+plural_most.p", "rb"))


candidate_adjs = []
for noun, props in noun2prop.items():
    candidate_adjs += props
candidate_adjs = list(set(candidate_adjs))

concreteness = {w: c / 5 for w, c in pickle.load(open("../data/word2concreteness.M.p", "rb")).items()}
all_words = list(concreteness.keys())

import random
prop2concretness = {}
for prop in tqdm(candidate_adjs):
    if prop in concreteness:
        prop2concretness[prop] = concreteness[prop]
    else:
        sims = []
        for word in all_words:
            sims.append((word, similar(word, prop)))
        sims.sort(key=lambda x: x[1], reverse=True)
        prop2concretness[prop] = concreteness[sims[0][0]]

prop2concretness_pred = pickle.load(open("/nlp/data/yueyang/prototypicality/predicted_adjective2concreteness.p", "rb"))

import random
noun2concretness = {}
for noun in tqdm(noun2prop):
    if noun in concreteness:
        noun2concretness[noun] = concreteness[noun]
    else:
        sims = []
        for word in all_words:
            sims.append((word, similar(word, noun)))
        sims.sort(key=lambda x: x[1], reverse=True)
        noun2concretness[noun] = concreteness[sims[0][0]]

noun2predicts = {}
lambs = []
for noun, c_scores in clip_scores.items():
    b_order = {p:i for i, p in enumerate(roberta_scores[noun])}
    c_order = {p:i for i, p in enumerate(c_scores)}
    combine_order = {}
    for prop, rank in c_order.items():
        # lamb = random.uniform(0, 1)
        lamb = prop2concretness_pred[prop] / 5
        # lambs.append(lamb)
        # lamb = 0.5
        combine_order[prop] = (1-lamb) * b_order[prop] + lamb * rank
        # combine_order[prop] = min(b_order[prop], rank)
    predicts = [(p, r) for p, r in combine_order.items()]
    predicts.sort(key=lambda x: x[0])
    predicts.sort(key=lambda x: x[1], reverse=False)
    noun2predicts[noun] = [pred[0] for pred in predicts]
    