import gzip
from collections import defaultdict
from tqdm import tqdm
import pickle
import os
import sys
sys.path.append('../..')


DATASET = "concept_properties" # "feature_norms", "memory_colors"

# def get_ngram():
#    noun2prop2count = {}
#    all_zip = [f for f in os.listdir("corpora/LDC/LDC2006T13/data/2gms/") if "gz" in f]
#    for file_name in tqdm(all_zip):
#       with gzip.open("corpora/LDC/LDC2006T13/data/2gms/" + file_name) as f:
#          bytecontents = f.read()
#       contents = bytecontents.decode("utf-8")
#       contents = contents.split("\n")
#       for content in contents:
#          s = content.strip().split("\t")
#          if len(s) == 2:
#                context_token, count = s
#                try:
#                   context, token = context_token.split(" ")
#                   if token not in noun2prop2count:
#                      noun2prop2count[token] = {context:int(count)}
#                   else: 
#                      noun2prop2count[token][context] = int(count)
#                except:
#                   continue
#    return noun2prop2count
                  
class NGram():
    def __init__(self, dataset, test=None):
      if dataset == 'concept_properties' and test:
         noun2prop = pickle.load(open(f"data/datasets/{dataset}/noun2property/noun2prop_test.p", "rb"))
      else:
         noun2prop = pickle.load(open(f"data/datasets/{dataset}/noun2property/noun2prop.p", "rb"))
      candidate_adjs = []

      for noun, props in noun2prop.items():
           candidate_adjs += props
      candidate_adjs = list(set(candidate_adjs))
      all_nouns = list(noun2prop.keys())
      noun2prop2count = pickle.load(open('models/ngram/noun2prop2count_ngram.p', 'rb'))
      self.noun2prop2count = noun2prop2count
      noun2predicts = {}
      for noun in tqdm(noun2prop):
         predicts = [(prop, count) for prop, count in noun2prop2count[noun].items()]
         predicts.sort(key=lambda x: x[0])
         predicts.sort(key=lambda x: x[1], reverse=True)
         noun2predicts[noun] = [pred[0] for pred in predicts]
      self.noun2predicts = noun2predicts