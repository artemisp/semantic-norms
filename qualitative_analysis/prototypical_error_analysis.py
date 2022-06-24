import pickle
import numpy as np
import sys
sys.path.append('..')
import eval
import random 
from collections import Counter

## Feature Norms
noun2prop2count = pickle.load(open('../models/ngram/noun2prop2count_ngram.p', 'rb'))
noun2prop = pickle.load(open('../data/datasets/feature_norms/noun2property/noun2prop.p', 'rb'))
prop2conc = pickle.load(open('../data/concreteness/word2concreteness.M.p', 'rb'))

all_prop = []
for _,prop in noun2prop.items():
    all_prop.extend(prop)
all_prop = list(set(all_prop))

prototypical_noun2prop = {}
lines = open('prorotypical_properties.prop', 'r').readlines()
for line in lines:
    line = line.strip().replace('.', '')
    noun = line.split('::')[0].strip()
    if noun not in noun2prop:
        continue
    prop = line.split(' ')[-1]
    if noun in prototypical_noun2prop:
        prototypical_noun2prop[noun].append(prop)
    else:
         prototypical_noun2prop[noun] = [prop]

for noun,proto in prototypical_noun2prop.items():
    for pp in proto:
        if noun in noun2prop and pp not in noun2prop[noun]:
            proto.remove(pp)
            prototypical_noun2prop[noun] = proto


non_prototypical_noun2prop = {}
for noun in noun2prop.keys():
    if noun not in prototypical_noun2prop:
        non_prototypical_noun2prop[noun] = noun2prop[noun]
    else:
        non_prototypical_noun2prop[noun] = set(noun2prop[noun]).difference(set(prototypical_noun2prop[noun]))



# print("Prototypical Concreteness")
# prototypical_concreteness = []
# prototypical_props = []
# for _, props in prototypical_noun2prop.items():
#     for p in props:
#         if p not in prototypical_props:
#             prototypical_props.append(p)
#             prototypical_concreteness.append(prop2conc[p])
# print(f"Mean:{np.mean(prototypical_concreteness)} Median:{np.median(prototypical_concreteness)},  Variance: {np.std(prototypical_concreteness)}")



# print("Non Prototypical Concreteness")
# non_prototypical_concreteness = []
# non_prototypical_props = []
# for _, props in non_prototypical_noun2prop.items():
#     for p in props: 
#         if p not in non_prototypical_props:
#             non_prototypical_props.append(p)
#             non_prototypical_concreteness.append(prop2conc[p])
# print(f"Mean:{np.mean(non_prototypical_concreteness)} Median:{np.median(non_prototypical_concreteness)}, Variance: {np.std(non_prototypical_concreteness)}")

dataset='feature_norms'
noun2prop = pickle.load(open('../data/datasets/{}/noun2property/noun2prop{}.p'.format(dataset, '_test' if dataset == 'concept_properties' else ''), "rb"))
gpt3_predicts = pickle.load(open(f'../output/output_{dataset}/gpt3_predicts.p', "rb"))
roberta_predicts = pickle.load(open(f'../output/output_{dataset}/roberta-large+singular_generally.p', "rb"))
bert_predicts = pickle.load(open(f'../output/output_{dataset}/bert-large-uncased+plural_most.p', "rb"))
gpt_predicts = pickle.load(open(f'../output/output_{dataset}/gpt2-large+plural_most.p', "rb"))
vilt_predicts = pickle.load(open(f'../output/output_{dataset}/vilt+plural+10.p', "rb"))
clip_predicts = pickle.load(open(f'../output/output_{dataset}/clip_scores.p', "rb"))
combined_predicts = pickle.load(open(f'../output/output_{dataset}/combine_scores.p', "rb"))
glove_predicts = pickle.load(open(f'../output/output_{dataset}/glove_noun2predicts.p', 'rb'))
random_predicts = {noun:random.sample(list(all_prop), len(all_prop)) for noun in noun2prop.keys()}
pred_combined_predicts = pickle.load(open(f'../output/output_{dataset}/pred_combined_scores.p', "rb"))


ngram_noun2predicts = {}
for noun, prop2coun in noun2prop2count.items():
    predicts = [(prop, count) for prop, count in prop2coun.items()]
    predicts += [(prop, 0) for prop in all_prop if prop not in prop2coun]
    predicts.sort(key=lambda x: x[0])
    predicts.sort(key=lambda x: x[1], reverse=True)
    ngram_noun2predicts[noun] = [pred[0] for pred in predicts]

model2predicts = {"random": random_predicts, "glove": glove_predicts, "ngram": ngram_noun2predicts, "bert": bert_predicts, "roberta": roberta_predicts, "gtp2": gpt_predicts, "gpt3": gpt3_predicts, "vilt": vilt_predicts, "clip": clip_predicts, "cem": combined_predicts, 'cem-pred': pred_combined_predicts}
model2predicts = {'cem-pred': pred_combined_predicts}


for model, noun2predicts in model2predicts.items():
    print(model)
    print("Only Some")
    eval.evaluate_acc(noun2predicts, non_prototypical_noun2prop, 5, PRINT=True)
    eval.evaluate_acc(noun2predicts, non_prototypical_noun2prop, 10, PRINT=True)
    eval.evaluate_recall(noun2predicts, non_prototypical_noun2prop, 5, PRINT=True)
    eval.evaluate_recall(noun2predicts, non_prototypical_noun2prop, 10, PRINT=True)
    if model != 'gpt3':
        eval.evaluate_rank(noun2predicts, non_prototypical_noun2prop, PRINT=True)
    print("Only All/Most")
    eval.evaluate_acc(noun2predicts, prototypical_noun2prop, 5, PRINT=True)
    eval.evaluate_acc(noun2predicts, prototypical_noun2prop, 10, PRINT=True)
    eval.evaluate_recall(noun2predicts, prototypical_noun2prop, 5, PRINT=True)
    eval.evaluate_recall(noun2predicts, prototypical_noun2prop, 10, PRINT=True)
    if model != 'gpt3':

       eval.evaluate_rank(noun2predicts, prototypical_noun2prop, PRINT=True)
    print("Some/All/Most")
    eval.evaluate_acc(noun2predicts, noun2prop, 5, PRINT=True)
    eval.evaluate_acc(noun2predicts, noun2prop, 10, PRINT=True)
    eval.evaluate_recall(noun2predicts, noun2prop, 5, PRINT=True)
    eval.evaluate_recall(noun2predicts, noun2prop, 10, PRINT=True)
    if model != 'gpt3':
       eval.evaluate_rank(noun2predicts, noun2prop, PRINT=True)

