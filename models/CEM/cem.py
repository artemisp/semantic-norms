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
    def __init__(self, dataset, model_concrete='clip', model_abstract='roberta', predicted_concreteness=False, recompute=False):
        
        noun2prop = pickle.load(open('data/datasets/{}/noun2property/noun2prop.p'.format(dataset), "rb"))
        candidate_adjs = []
        for noun, props in noun2prop.items():
            candidate_adjs += props
        candidate_adjs = list(set(candidate_adjs))

        if not recompute:
            if model_concrete == 'clip':
                concrete_scores =  pickle.load(open(f"data/datasets/{dataset}/CLIP/clip_scores.p", "rb"))
            
            if model_abstract == 'roberta':
                abstract_scores = pickle.load(open(f"output/output_{dataset}/roberta-large+singular_generally.p", "rb"))

        else:
            if model_concrete == 'clip':
                from models.CLIP.clip_openai import CLIP
                concrete_scores = CLIP(dataset).noun2predicts
            
            if model_abstract == 'roberta':
                from models.lm.mlm_multitok import LM
                abstract_scores =  LM('roberta', dataset).prompt2noun2predicts['singular_generally']
            
        if predicted_concreteness:
            prop2concretness = {w: c / 5 for w, c in pickle.load(open("data/concreteness/predicted_adjective2concreteness.p", "rb")).items()}
        else:
            concreteness = {w: c / 5 for w, c in pickle.load(open("data/concreteness/word2concreteness.M.p", "rb")).items()}
            all_words = list(concreteness.keys())

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
                if type(prop) == tuple:
                    prop = prop[0]
                try:
                    lamb = prop2concretness[prop]
                except:
                    from pdb import set_trace; set_trace()
                # lambs.append(lamb)
                # lamb = 0.5
                combine_order[prop] = (1-lamb) * b_order[prop] + lamb * rank
                # combine_order[prop] = max(b_order[prop], rank)
            predicts = [(p, r) for p, r in combine_order.items()]
            predicts.sort(key=lambda x: x[0])
            predicts.sort(key=lambda x: x[1], reverse=False)
            noun2predicts[noun] = [pred[0] for pred in predicts]
        self.noun2predicts = noun2predicts

# def lcs(X, Y, m=5, n=5):
 
#     if m == 0 or n == 0:
#        return 0;
#     elif X[m-1] == Y[n-1]:
#        return 1 + lcs(X, Y, m-1, n-1);
#     else:
#        return max(lcs(X, Y, m, n-1), lcs(X, Y, m-1, n));

# gold  = CEM('feature_norms')
# pred = CEM('feature_norms', predicted_concreteness=True)
# similarity  = []
# for noun, preds in gold.noun2predicts.items():
#     similarity.append(lcs(preds[:5], pred.noun2predicts[noun][:5]))
    
#     # similarity.append(len(set(preds[:5]).intersection(set(pred.noun2predicts[noun][:5]))))

# from pdb import set_trace; set_trace()