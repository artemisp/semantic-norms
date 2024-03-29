import pickle 
import random
import numpy as np

for dataset in ['concept_properties', 'feature_norms']:
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

    objects = random.sample(noun2prop.keys(), 20)
    print(f"{dataset}")
    for object in objects:
        if object not in noun2sorted_images:
            continue
        out = "\\multirow{4}{*}{"+object+"} & \\multirow{4}{*}\{\\cincludegraphics[width=1.8cm]{"+noun2sorted_images[object][0]+"}"
        for model, predicts in model2predicts.items():
            if model.lower() == 'roberta':
                model_ = f'& {model}'
            else:
                model_ = f'& & {model}'
            if model.lower() in ['clip']:
                 out+=f"{model_} {','.join([p[0] for p in predicts[object][:5]])} \\\\ \n"
            else:
                out+=f"{model_} {','.join(predicts[object][:5])} \\\\ \n"
        print(out)

