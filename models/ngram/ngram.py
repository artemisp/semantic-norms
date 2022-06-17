import gzip
from collections import defaultdict
from tqdm import tqdm
import pickle
import os


DATASET = "concept_properties" # "feature_norms", "memory_colors"


noun2prop = pickle.load(open(f"../data/datasets/{DATASET}/noun2property/noun2prop.p", "rb"))
candidate_adjs = []

for noun, props in noun2prop.items():
    candidate_adjs += props
candidate_adjs = list(set(candidate_adjs))
all_nouns = list(noun2prop.keys())

noun2prop2count = pickle.load(open('noun2prop2count_ngram.p', 'rb'))

# all_zip = [f for f in os.listdir("/nlp/data/corpora/LDC/LDC2006T13/data/2gms/") if "gz" in f]
# for file_name in tqdm(all_zip):
#     with gzip.open("/nlp/data/corpora/LDC/LDC2006T13/data/2gms/" + file_name) as f:
#         bytecontents = f.read()
#     contents = bytecontents.decode("utf-8")
#     contents = contents.split("\n")
#     for content in contents:
#         s = content.strip().split("\t")
#         if len(s) == 2:
#             context_token, count = s
#             context, token = context_token.split(" ")
#             if token not in noun2prop2count:
#                  noun2prop2count[token] = {context:int(count)}
#             else: 
#                 noun2prop2count[token][context] = int(count)

noun2predicts = {}
for noun, prop2coun in noun2prop2count.items():
    predicts = [(prop, count) for prop, count in prop2coun.items()]
    predicts.sort(key=lambda x: x[0])
    predicts.sort(key=lambda x: x[1], reverse=True)
    noun2predicts[noun] = [pred[0] for pred in predicts]

acc_1 = eval.evaluate_acc(noun2predicts, noun2prop, 1, True)
acc_5 = eval.evaluate_acc(noun2predicts, noun2prop, 5, True)
r_5 = eval.evaluate_recall(noun2predicts, noun2prop, 5, True)
r_10 = eval.evaluate_recall(noun2predicts, noun2prop, 10, True)
mrr = eval.evaluate_rank(noun2predicts, noun2prop, True)