import pickle 
import random

for dataset in ['cslb', 'mrd']:
    if dataset == 'cslb':
        noun2sorted_images = pickle.load(open('/nlp/data/yueyang/prototypicality/CSLB/data/noun2sorted_images.p', 'rb'))
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
        noun2sorted_images = pickle.load(open('/nlp/data/yueyang/prototypicality/MRD/data/noun2sorted_images.p', 'rb'))
        noun2prop = pickle.load(open("/nlp/data/yueyang/prototypicality/MRD/data/MRD/MRD_noun2prop.p", "rb"))
        gpt3_predicts = pickle.load(open("/nlp/data/yueyang/prototypicality/MRD/model/GPT3/gpt3_ten_adjs.p", "rb"))
        roberta_predicts = pickle.load(open("/nlp/data/yueyang/prototypicality/MRD/output/roberta-large+singular_generally.p", "rb"))
        clip_predicts = pickle.load(open("/nlp/data/yueyang/prototypicality/MRD/preprocess/clip_scores_l14.p", "rb"))
        bert_predicts = pickle.load(open("/nlp/data/yueyang/prototypicality/MRD/output/bert-large-uncased+plural_most.p", "rb"))
        gpt_predicts = pickle.load(open("/nlp/data/yueyang/prototypicality/MRD/output/gpt2-large+plural_most.p", "rb"))
        vilt_predicts = pickle.load(open("/nlp/data/yueyang/prototypicality/MRD/output/vilt+plural+10.p", "rb"))
        combined_predicts = pickle.load(open("/nlp/data/yueyang/prototypicality/MRD/preprocess/combine_scores.p", "rb"))

    model2predicts = {"bert": bert_predicts, "roberta": roberta_predicts, "gtp2": gpt_predicts, "gpt3": gpt3_predicts, "vilt": vilt_predicts, "clip": clip_predicts, "cem": combined_predicts}
    model2predicts = { "RoBERTa": roberta_predicts,"CLIP": clip_predicts, "CEM": combined_predicts, "GPT-3 ": gpt3_predicts}

    objects = random.sample(noun2prop.keys(), 20)
    print(f"{dataset}")
    for object in objects:
        out = "\\multirow{4}{*}{"+object+"} & \\multirow{4}{*}\{\\cincludegraphics[width=1.8cm]{"+noun2sorted_images[object][0]+"}"
        for model, predicts in model2predicts.items():
            if model.lower() == 'roberta':
                model = f'& {model}'
            else:
                model = f'& & {model}'
            out+=f"{model} {','.join(predicts[object][:5])} \\\\ \n"
        print(out)

