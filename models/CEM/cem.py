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

# noun2prop = pickle.load(open(f"../data/datasets/{DATASET}/noun2property/noun2prop.p", "rb"))
# gpt3_scores = pickle.load(open(f"../data/datasets/{DATASET}/GPT3/gpt3_predicts.txt", "rb"))
# roberta_scores = pickle.load(open(f"../output/output_{DATASET}/roberta-large+singular_generally.p", "rb"))
# bert_scores = pickle.load(open(f"../output/output_{DATASET}/bert-large-uncased+plural_most.p", "rb"))
# vilt_scores = pickle.load(open(f"../output/output_{DATASET}/vilt+plural+10.p", "rb"))
# clip_scores = pickle.load(open(f"../data/datasets/{DATASET}/CLIP/clip_scores.p", "rb"))
# combined_scores = pickle.load(open(f"../data/datasets/{DATASET}/CEM/combine_scores.p", "rb"))
# ngram_scores = pickle.load(open("../data/datasets/feature_norms/ngram_scores.p", "rb"))
# gpt_scores = pickle.load(open(f"../output/output_{DATASET}/gpt2-large+plural_most.p", "rb"))



class CEM():
    def __init__(self, dataset, model_concrete='clip', model_abstract='roberta', predicted_concreteness=False, recompute=False):
        
        noun2prop = pickle.load(open(f'../../data/datasets/{dataset}/noun2property/noun2prop.p', "rb"))
        candidate_adjs = []
        for noun, props in noun2prop.items():
            candidate_adjs += props
        candidate_adjs = list(set(candidate_adjs))

        if not recompute:
            if model_concrete == 'clip':
                concrete_scores =  pickle.load(open(f"../../data/datasets/{dataset}/CLIP/clip_scores.p", "rb"))
            
            if model_abstract == 'roberta':
                abstract_scores = pickle.load(open(f"../../output/output_{dataset}/roberta-large+singular_generally.p", "rb"))

        else:
            if model_concrete == 'clip':
                from models.CLIP.clip_openai import CLIP
                concrete_scores = CLIP(dataset).noun2predicts
            
            if model_abstract == 'roberta':
                from models.lm.mlm_multitok import LM
                abstract_scores =  LM('roberta', dataset).prompt2noun2predicts['singular_generally']
            
        if predicted_concreteness:
            prop2concretness = {w: c / 5 for w, c in pickle.load(open("../../data/concreteness/predicted_adjective2concreteness.p", "rb")).items()}
        else:
            concreteness = {w: c / 5 for w, c in pickle.load(open("../../data/concreteness/word2concreteness.M.p", "rb")).items()}
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


        noun2predicts = {}
        lambs = []
        for noun, c_scores in concrete_scores.items():
            b_order = {p:i for i, p in enumerate(abstract_scores[noun])}
            c_order = {p:i for i, p in enumerate(c_scores)}
            combine_order = {}
            for prop, rank in c_order.items():
                # lamb = random.uniform(0, 1)
                prop = prop[0]
                lamb = prop2concretness[prop]
                # lambs.append(lamb)
                # lamb = 0.5
                combine_order[prop] = (1-lamb) * b_order[prop] + lamb * rank
                # combine_order[prop] = max(b_order[prop], rank)
            predicts = [(p, r) for p, r in combine_order.items()]
            predicts.sort(key=lambda x: x[0])
            predicts.sort(key=lambda x: x[1], reverse=False)
            noun2predicts[noun] = [pred[0] for pred in predicts]
        self.noun2predicts = noun2predicts
    
        eval.evaluate_recall(noun2predicts, noun2prop, K=5, PRINT=True)

CEM('feature_norms',predicted_concreteness=False)