import pickle
from collections import Counter
from transformers import RobertaTokenizer, BertTokenizer,CLIPTokenizer, OpenAIGPTTokenizer, GPT2Tokenizer
import sys
sys.path.append('..')
import eval

tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
bert_tok = BertTokenizer.from_pretrained('bert-large-uncased')
clip_tok = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
gpt_tok = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
gpt2_tok = GPT2Tokenizer.from_pretrained('openai-gpt')
dataset = 'cslb'

for dataset in ['concept_properties', 'feature_norms']:
    noun2sorted_images = pickle.load(open(f'../data/datasets/{dataset}/images/noun2sorted_images.p', 'rb'))
    noun2prop = pickle.load(open('../data/datasets/{}/noun2property/noun2prop{}.p'.format(dataset, '_test' if dataset == 'concept_properties' else ''), "rb"))
    gpt3_predicts = pickle.load(open(f'../output/output_{dataset}/gpt3_predicts.p', "rb"))
    roberta_predicts = pickle.load(open(f'../output/output_{dataset}/roberta-large+singular_generally.p', "rb"))
    bert_predicts = pickle.load(open(f'../output/output_{dataset}/bert-large-uncased+plural_most.p', "rb"))
    gpt_predicts = pickle.load(open(f'../output/output_{dataset}/gpt2-large+plural_most.p', "rb"))
    vilt_predicts = pickle.load(open(f'../output/output_{dataset}/vilt+plural+10.p', "rb"))
    clip_predicts = pickle.load(open(f'../output/output_{dataset}/clip_scores.p', "rb"))
    combined_predicts = pickle.load(open(f'../output/output_{dataset}/combine_predicts.p', "rb"))
    pred_combined_predicts = pickle.load(open(f'../output/output_{dataset}/pred_combined_scores.p', "rb"))



    model2predicts = {"bert": bert_predicts, "roberta": roberta_predicts, "gtp2": gpt_predicts, "gpt3": gpt3_predicts, "vilt": vilt_predicts, "clip": clip_predicts, "cem": combined_predicts, "cem-pred": pred_combined_predicts}
    model2tok = {"bert": bert_tok, "roberta": tokenizer, "gtp2": gpt2_tok, "gpt3": gpt_tok, "vilt": bert_tok, "clip": clip_tok, "cem": clip_tok, "cem-pred": clip_tok}


    print("Dataset: {}\n".format(dataset))
    for model,predicts in model2predicts.items():
        print("Model:{}\n".format(model))
        multitok_noun2prop = {}
        for noun,props in noun2prop.items():
            multitok_noun2prop[noun] = []
            for p in props:
                if len(model2tok[model](p)['input_ids']) > 3:
                    multitok_noun2prop[noun].append(p)
        
        print("# Multitok:{}\n".format(sum([len(v) for v in multitok_noun2prop.values()])))
        
        eval.evaluate_acc(predicts, multitok_noun2prop, 5, PRINT=True)
        eval.evaluate_acc(predicts, multitok_noun2prop, 10, PRINT=True)
        eval.evaluate_recall(predicts, multitok_noun2prop, 5, PRINT=True)
        eval.evaluate_recall(predicts, multitok_noun2prop, 10, PRINT=True)
        if model != 'gpt3':
            eval.evaluate_rank(predicts, multitok_noun2prop, PRINT=True)
