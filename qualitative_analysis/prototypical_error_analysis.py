import pickle
import numpy as np
import sys
sys.path.append('/nlp/data/yueyang/prototypicality/qualitative_results')
import eval
import random 
from collections import Counter

## MRD
noun2prop2count = pickle.load(open('/nlp/data/yueyang/prototypicality/MRD/model/ngram/MRD_ngram_count.p', 'rb'))
noun2prop = pickle.load(open('/nlp/data/yueyang/prototypicality/MRD/data/MRD/MRD_noun2prop.p', 'rb'))
prop2conc = pickle.load(open('/nlp/data/yueyang/prototypicality/MRD/data/MRD_prop2concreteness.p', 'rb'))

all_prop = []
for _,prop in noun2prop.items():
    all_prop.extend(prop)
all_prop = list(set(all_prop))

prototypical_noun2prop = {}
lines = open('/nlp/data/yueyang/prototypicality/qualitative_results/prorotypical_properties.prop', 'r').readlines()
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

mcrae_lines = open('/nlp/data/yueyang/prototypicality/qualitative_results/mcrae-quantified.txt', 'r').readlines()
mcrae_quant = {noun:{} for noun in noun2prop.keys()}
for line in mcrae_lines:
    split_line = line.split()
    if split_line[1] in noun2prop.keys() and split_line[2].split('_')[-1] in all_prop:
        mcrae_quant[split_line[1]][split_line[2].split('_')[-1]] = split_line[-3:]

all_all_all = []
all_all_most = []
most_most_all = []
most_most_some = []
some_some_x = []
most_most_x = []
some_some_some =[]
for noun,props in mcrae_quant.items():
    for p in props:
        c = Counter(mcrae_quant[noun][p])
        if c['all']  == 3:
            all_all_all.append(p)
        if c['some'] == 3:
             some_some_some.append(p)
        elif c['all'] == 2 and c['most'] == 1:
            all_all_most.append(p)
        elif c['most'] == 2 and 'some' in c:
            most_most_some.append(p)
        elif c['most'] == 2 and 'all' in c:
            most_most_all.append(p)
        elif c['some'] == 2 and ('few' not in c or 'no' not in c):
            some_some_x.append(p)
        else:
            print(c)
    pass

l = {'all_all_all':all_all_all,'some_some_some':some_some_some, 'all_all_most':all_all_most, 'most_most_all':most_most_all, 'most_most_some': most_most_some,'some_some_x':some_some_x, 'most_most_x':most_most_x}
for n,pr in l.items():
    conc = []
    if len(pr) == 0:
        continue
    for p in pr:
        conc.append(prop2conc[p])
    print(n)
    print(np.median(conc))




print("Prototypical Concreteness")
prototypical_concreteness = []
prototypical_props = []
for _, props in prototypical_noun2prop.items():
    for p in props:
        if p not in prototypical_props:
            prototypical_props.append(p)
            prototypical_concreteness.append(prop2conc[p])
print(f"Mean:{np.mean(prototypical_concreteness)} Median:{np.median(prototypical_concreteness)},  Variance: {np.std(prototypical_concreteness)}")



print("Non Prototypical Concreteness")
non_prototypical_concreteness = []
non_prototypical_props = []
for _, props in non_prototypical_noun2prop.items():
    for p in props: 
        if p not in non_prototypical_props:
            non_prototypical_props.append(p)
            non_prototypical_concreteness.append(prop2conc[p])
print(f"Mean:{np.mean(non_prototypical_concreteness)} Median:{np.median(non_prototypical_concreteness)}, Variance: {np.std(non_prototypical_concreteness)}")

from pdb import set_trace; set_trace()

gpt3_predicts = pickle.load(open("/nlp/data/yueyang/prototypicality/MRD/model/GPT3/gpt3_ten_adjs.p", "rb"))
roberta_predicts = pickle.load(open("/nlp/data/yueyang/prototypicality/MRD/output/roberta-large+singular_generally.p", "rb"))
clip_predicts = pickle.load(open("/nlp/data/yueyang/prototypicality/MRD/preprocess/clip_scores_l14.p", "rb"))
bert_predicts = pickle.load(open("/nlp/data/yueyang/prototypicality/MRD/output/bert-large-uncased+plural_most.p", "rb"))
gpt_predicts = pickle.load(open("/nlp/data/yueyang/prototypicality/MRD/output/gpt2-large+plural_most.p", "rb"))
vilt_predicts = pickle.load(open("/nlp/data/yueyang/prototypicality/MRD/output/vilt+plural+10.p", "rb"))
combined_predicts = pickle.load(open("/nlp/data/yueyang/prototypicality/MRD/preprocess/combine_scores.p", "rb"))
glove_predicts = pickle.load(open('/nlp/data/yueyang/prototypicality/MRD/model/word2vec/glove_noun2predicts.p', 'rb'))
random_predicts = {noun:random.sample(list(all_prop), len(all_prop)) for noun in noun2prop.keys()}

ngram_noun2predicts = {}
for noun, prop2coun in noun2prop2count.items():
    predicts = [(prop, count) for prop, count in prop2coun.items()]
    predicts += [(prop, 0) for prop in all_prop if prop not in prop2coun]
    predicts.sort(key=lambda x: x[0])
    predicts.sort(key=lambda x: x[1], reverse=True)
    ngram_noun2predicts[noun] = [pred[0] for pred in predicts]

model2predicts = {"random": random_predicts, "glove": glove_predicts, "ngram": ngram_noun2predicts, "bert": bert_predicts, "roberta": roberta_predicts, "gtp2": gpt_predicts, "gpt3": gpt3_predicts, "vilt": vilt_predicts, "clip": clip_predicts, "cem": combined_predicts}

from pdb import set_trace; set_trace()

# import gzip
# from collections import defaultdict

# def load_freq_counts(frequency_fn="/nlp/data/corpora/LDC/LDC2006T13/data/1gms/vocab.gz"):                     

#         with gzip.open(frequency_fn) as f:
#                 bytecontents = f.read()
#         contents = bytecontents.decode("utf-8")
#         contents = contents.split("\n")
#         freq_counts = defaultdict(int) 

#         for tokencount in contents:
#                 s = tokencount.strip().split("\t")
#                 if len(s) == 2:
#                         token, count = s
#                         freq_counts[token] = int(count)

#         return freq_counts

# freq_counts = load_freq_counts()

# noun2incorrect = {}
# noun2correct = {}
# for noun, prop in noun2prop.items():
#     noun2incorrect[noun] = []
#     noun2correct[noun] = []
#     for p in prop:
#         if p not in ngram_noun2predicts[noun]:
#             noun2incorrect[noun].append(p)
#         else:
#             noun2correct[noun].append(p)


# concreteness_incorrect = []

# for incorrect in noun2incorrect.values():
#     for p in incorrect:
#         concreteness_incorrect.append(prop2conc[p])

# print("Mean incorrect concreteness")
# print(np.mean(concreteness_incorrect))

# concreteness_correct = []
# for correct in noun2correct.values():
#     for p in correct:
#         concreteness_correct.append(prop2conc[p])

# print("Mean correct concreteness")
# print(np.mean(concreteness_correct))


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

from pdb import set_trace; set_trace()


