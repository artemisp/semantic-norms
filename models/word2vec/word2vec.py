import sys
sys.path.insert(1, '../..')
import eval
import pickle
import random
from tqdm import tqdm
import torch as th
import numpy as np
import os
from sentence_transformers import SentenceTransformer, util

DATASET = "concept_properties" # "feature_norms", "memory_colors"

model = SentenceTransformer('average_word_embeddings_glove.6B.300d', device = "cuda:0")

def get_text_embeddings(sentences):
    return model.encode(sentences, batch_size = 32)

if __name__ == "__main__":
    noun2prop = pickle.load(open("/nlp/data/yueyang/prototypicality/clean/data/datasets/{DATASET}/noun2property/noun2prop.p", "rb"))
    candidate_adjs = []
    for noun, props in noun2prop.items():
        candidate_adjs += props
    candidate_adjs = list(set(candidate_adjs))
    adj_embeddings = get_text_embeddings(candidate_adjs)

    all_nouns = list(noun2prop.keys())
    noun2predicts = {}
    for noun in tqdm(all_nouns):
        noun_embeddings = model.encode([noun])
        cosine_scores = util.cos_sim(adj_embeddings, noun_embeddings)
        cosine_scores_avg = th.sum(cosine_scores, dim = 1)
        sorted_prop_index = th.argsort(cosine_scores_avg, descending=True)
        outputs = [candidate_adjs[ind] for ind in sorted_prop_index]
        noun2predicts[noun] = outputs
        # random.shuffle(candidate_adjs)
        # noun2predicts[noun] = candidate_adjs
    acc_1 = eval.evaluate_acc(noun2predicts, noun2prop, 1, True)
    acc_5 = eval.evaluate_acc(noun2predicts, noun2prop, 5, True)
    r_5 = eval.evaluate_recall(noun2predicts, noun2prop, 5, True)
    r_10 = eval.evaluate_recall(noun2predicts, noun2prop, 10, True)
    mrr = eval.evaluate_rank(noun2predicts, noun2prop, True)