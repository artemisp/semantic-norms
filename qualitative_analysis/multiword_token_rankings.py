import pickle
from collections import Counter
from transformers import RobertaTokenizer, BertTokenizer,CLIPTokenizer, OpenAIGPTTokenizer, GPT2Tokenizer
import sys
sys.path.append('../..')
import eval

tokenizer = RobertaTokenizer.from_pretrained('roberta-large', cache_dir='/nlp/data/artemisp/huggingface')
bert_tok = BertTokenizer.from_pretrained('bert-large-uncased', cache_dir='/nlp/data/artemisp/huggingface')
clip_tok = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32', cache_dir='/nlp/data/artemisp/huggingface')
gpt_tok = OpenAIGPTTokenizer.from_pretrained('openai-gpt', cache='/nlp/data/artemisp/huggingface')
gpt2_tok = GPT2Tokenizer.from_pretrained('openai-gpt', cache='/nlp/data/artemisp/huggingface')
dataset = 'cslb'
k = 5

for dataset in ['cslb', 'mrd']:
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

    model2predicts = {"bert": bert_predicts, "roberta": roberta_predicts, "gtp2": gpt_predicts, "gpt3": gpt3_predicts, "vilt": vilt_predicts, "clip": clip_predicts, "cem": combined_predicts}
    model2tok = {"bert": bert_tok, "roberta": tokenizer, "gtp2": gpt2_tok, "gpt3": gpt_tok, "vilt": bert_tok, "clip": clip_tok, "cem": clip_tok}

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
