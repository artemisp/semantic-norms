import pickle
from collections import Counter
import numpy as np
from collections import defaultdict
import gzip
import matplotlib.pyplot as plt

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

dataset = 'mrd'
k = 5

for dataset in ['cslb', 'mrd']:
    model2meannouns = {}
    for k in [3, 4, 5]:
        if dataset == 'cslb':
            noun2prop = pickle.load(open("/nlp/data/yueyang/prototypicality/CSLB/data/CSLB_noun2prop.p", "rb"))
            noun2prop_train = pickle.load(open("/nlp/data/yueyang/prototypicality/CSLB/data/CSLB_noun2prop_train.p", "rb"))
            noun2prop_test = pickle.load(open("/nlp/data/yueyang/prototypicality/CSLB/data/CSLB_noun2prop_test.p", "rb"))
            gpt3_predicts = pickle.load(open("/nlp/data/yueyang/prototypicality/CSLB/model/GPT3/gpt3_predicts_CSLB", "rb"))
            roberta_predicts = pickle.load(open("/nlp/data/yueyang/prototypicality/CSLB/output/roberta-large+singular_generally.p", "rb"))
            bert_predicts = pickle.load(open("/nlp/data/yueyang/prototypicality/CSLB/output/bert-large-uncased+plural_most.p", "rb"))
            gpt_predicts = pickle.load(open("/nlp/data/yueyang/prototypicality/CSLB/output/gpt2-large+plural_most.p", "rb"))
            vilt_predicts = pickle.load(open("/nlp/data/yueyang/prototypicality/CSLB/output/vilt+plural+10.p", "rb"))
            clip_predicts = pickle.load(open("/nlp/data/yueyang/prototypicality/CSLB/preprocess/clip_scores.p", "rb"))
            combined_predicts = pickle.load(open("/nlp/data/yueyang/prototypicality/CSLB/preprocess/combine_predicts.p", "rb"))

        else:
            noun2prop = pickle.load(open("/nlp/data/yueyang/prototypicality/MRD/data/MRD/MRD_noun2prop.p", "rb"))
            gpt3_predicts = pickle.load(open("/nlp/data/yueyang/prototypicality/MRD/model/GPT3/gpt3_ten_adjs.p", "rb"))
            roberta_predicts = pickle.load(open("/nlp/data/yueyang/prototypicality/MRD/output/roberta-large+singular_generally.p", "rb"))
            clip_predicts = pickle.load(open("/nlp/data/yueyang/prototypicality/MRD/preprocess/clip_scores_l14.p", "rb"))
            bert_predicts = pickle.load(open("/nlp/data/yueyang/prototypicality/MRD/output/bert-large-uncased+plural_most.p", "rb"))
            gpt_predicts = pickle.load(open("/nlp/data/yueyang/prototypicality/MRD/output/gpt2-large+plural_most.p", "rb"))
            vilt_predicts = pickle.load(open("/nlp/data/yueyang/prototypicality/MRD/output/vilt+plural+10.p", "rb"))
            combined_predicts = pickle.load(open("/nlp/data/yueyang/prototypicality/MRD/preprocess/combine_scores.p", "rb"))

        model2predicts = {"GPT3":gpt3_predicts, "RoBERTa":roberta_predicts, "BERT":bert_predicts, "GPT2": gpt_predicts, "CLIP": clip_predicts, "ViLT": vilt_predicts, "CEM": combined_predicts}

        print(k)
        print(dataset)
        for model,predicts in model2predicts.items():
            preds2noun = {}
            mean_nouns = []
            for n,v in predicts.items():
                if str(v[:k]) in preds2noun:
                    preds2noun[str(v[:k])].append(n)
                else:
                    preds2noun[str(v[:k])] = [n]
            print(f"{model} {np.mean([len(v) for k,v in preds2noun.items()])}")
            if model not in model2meannouns:
                 model2meannouns[model] = [np.sum([len(v) for k,v in preds2noun.items() if len(v) > 1])]
            else:
                model2meannouns[model].append(np.sum([len(v) for k,v in preds2noun.items() if len(v) > 1]))

    print(model2meannouns)
    # for model, meannouns in model2meannouns.items():
    #     plt.plot([1,3,5], model2meannouns[model], label=model)

    # plt.legend()
    # plt.xlabel('k')
    # plt.ylabel('mean duplicate predictions')
    # plt.savefig(f'/nlp/data/yueyang/prototypicality/qualitative_results/mean_dup_{dataset}.png')

            
        # frequencies = []
        # for noun,prop in vilt_predicts.items():
        #     for p in prop[:5]:
        #         frequencies.append(freq_counts[p])
        
        # print(np.mean(frequencies)/1000000)







        # combined_top5 = Counter([v_ for v in combined_predicts.values() for v_ in v[:k]])
        # combined_top5 =  sorted(combined_top5.items(), key=lambda x: x[1], reverse=True)
        # combined_top5_freq_counts = Counter([k[1] for k in combined_top5])

        # roberta_top5 = Counter([v_ for v in roberta_predicts.values() for v_ in v[:k]])
        # roberta_top5 =  sorted(roberta_top5.items(), key=lambda x: x[1], reverse=True)
        # roberta_top5_freq_counts = Counter([k[1] for k in roberta_top5])

        # clip_top5 = Counter([v_ for v in clip_predicts.values() for v_ in v[:k]])
        # clip_top5 =  sorted(clip_top5.items(), key=lambda x: x[1], reverse=True)
        # clip_top5_freq_counts = Counter([k[1] for k in clip_top5])

        # gpt_top5 = Counter([v_ for v in gpt3_predicts.values() for v_ in v[:k]])
        # gpt_top5 =  sorted(gpt_top5.items(), key=lambda x: x[1], reverse=True)
        # gpt_top5_freq_counts = Counter([k[1] for k in gpt_top5])

        # print(f"k: {k} data: {dataset}")
        # print(f'CEM {combined_top5[0][0]} {combined_top5[0][1]} {len(combined_top5)}')
        # print(f'Roberta {roberta_top5[0][0]} {roberta_top5[0][1]} {len(roberta_top5)}')
        # print(f'CLIP {clip_top5[0][0]} {clip_top5[0][1]} {len(clip_top5)}')
        # print(f'GPT {gpt_top5[0][0]} {gpt_top5[0][1]} {len(gpt_top5)}')
        # print('\n\n')


        # import matplotlib.pyplot as plt

        # plt.xlabel(f'Property Frequency in top {k}')
        # plt.ylabel('# Properties')

        # plt.plot(combined_top5_freq_counts.keys(),combined_top5_freq_counts.values(), label='CEM')
        # plt.plot(roberta_top5_freq_counts.keys(),roberta_top5_freq_counts.values(),label='RoBERTa')
        # plt.plot(clip_top5_freq_counts.keys(),clip_top5_freq_counts.values(),label='CLIP')
        # # plt.plot(gpt_top5_freq_counts.keys(), gpt_top5_freq_counts.values(),label='GPT3')
        # plt.legend()

        # plt.savefig(f'/nlp/data/yueyang/prototypicality/qualitative_results/top{k}_freq_histograms_{dataset}.png')
        # plt.cla()
        # # from pdb import set_trace; set_trace()
        # import random; keys = random.sample(gpt_predicts.keys(), 10)

        # for k in keys:
        #     print(f"\n {k}")
        #     for model, preds in {"CEM": combined_predicts, "GPT": gpt3_predicts, "RoBERTa": roberta_predicts, "CLIP": clip_predicts}.items():
        #         print(f"{model}: {preds[k][0]}, {preds[k][1]}, {preds[k][2]}, {preds[k][3]}, {preds[k][4]}")



