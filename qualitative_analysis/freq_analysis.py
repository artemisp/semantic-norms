import pickle 
import random
import numpy as np
from collections import defaultdict
import gzip

def load_freq_counts(frequency_fn="/nlp/data/corpora/LDC/LDC2006T13/data/1gms/vocab.gz"):                     

        with gzip.open(frequency_fn) as f:
                bytecontents = f.read()
        contents = bytecontents.decode("utf-8")
        contents = contents.split("\n")
        freq_counts = defaultdict(int) 

        for tokencount in contents:
                s = tokencount.strip().split("\t")
                if len(s) == 2:
                        token, count = s
                        freq_counts[token] = int(count)

        return freq_counts

freq_counts = load_freq_counts()


for dataset in ['feature_norms']:
    noun2sorted_images = pickle.load(open(f'../data/datasets/{dataset}/images/noun2sorted_images.p', 'rb'))
    noun2prop = pickle.load(open('../data/datasets/{}/noun2property/noun2prop{}.p'.format(dataset, '_test' if dataset == 'concept_properties' else ''), "rb"))
    gpt3_predicts = pickle.load(open(f'../output/output_{dataset}/gpt3_predicts.p', "rb"))
    roberta_predicts = pickle.load(open(f'../output/output_{dataset}/roberta-large+singular_generally.p', "rb"))
    bert_predicts = pickle.load(open(f'../output/output_{dataset}/bert-large-uncased+plural_most.p', "rb"))
    gpt_predicts = pickle.load(open(f'../output/output_{dataset}/gpt2-large+plural_most.p', "rb"))
    vilt_predicts = pickle.load(open(f'../output/output_{dataset}/vilt+plural+10.p', "rb"))
    clip_predicts = pickle.load(open(f'../output/output_{dataset}/clip_scores.p', "rb"))
    combined_predicts = pickle.load(open(f'../output/output_{dataset}/combine_scores.p', "rb"))
    pred_combined_predicts = pickle.load(open(f'../output/output_{dataset}/pred_combined_scores.p', "rb"))

    model2predicts = { "RoBERTa": roberta_predicts,"CLIP": clip_predicts, "CEM": combined_predicts, "CEM-Pred": pred_combined_predicts, "GPT-3 ": gpt3_predicts}

    noun2prop2count = pickle.load(open('/nlp/data/yueyang/prototypicality/semantic_norms/semantic-norms/models/ngram/noun2prop2count_ngram_.p', 'rb'))
    for model, predicts in model2predicts.items():
        model_1 = []
        model_2 = []
        print(model)
        for noun, pred in predicts.items():
            for p in pred[:5]:
                model_1.append(freq_counts[p])
                model_2.append(noun2prop2count[noun][p] if p in noun2prop2count[noun] else 0)
        print(np.mean(model_1))
        print(np.mean(model_2))
    continue