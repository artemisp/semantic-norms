import argparse
import sys
sys.path.append('.')
import pickle
import eval


parser = argparse.ArgumentParser(description='Provide the dataset and model on which you want to run the experiments on.')
parser.add_argument('--dataset',type=str, default='feature_norms', help='Options: concept_properties, feature_norms, memory_colors')
parser.add_argument('--model', type=str, default='roberta', help='Options: random, glove, ngram, bert, roberta, gpt2, gpt3, vilt, clip, cem, cem-pred')

args = parser.parse_args()

if args.dataset not in ['concept_properties', 'feature_norms', 'memory_colors']:
    print("DATASET not recognized. Select one of: concept_properties, feature_norms, or memory_colors")
    exit(0)

if args.model not in ['random', 'glove', 'ngram', 'bert', 'roberta', 'gpt2', 'gpt3', 'vilt', 'clip', 'cem', 'cem-pred']:
    print("Model not recognized. Select one of: random, glove, ngram, bert, roberta, gpt2, gpt3, vilt, clip, cem, cem-pred")
    exit(0)

DATASET = args.dataset
noun2prop = pickle.load(open(f'../data/datasets/{DATASET}/noun2property/noun2prop.p', 'rb'))

if args.model == 'random':
    import random
    import numpy as np
    all_props = list(set([v_ for v in noun2prop.values() for v_ in v ]))
    noun2predicts = {}
    for noun in noun2prop:
        noun2predicts[noun] = random.sample(all_props, len(all_props))

elif args.model == 'glove':
    from models.word2vec.word2vec import Word2Vec
    noun2predicts = Word2Vec(DATASET).noun2predicts

elif args.model == 'ngram':
    from models.ngram.ngram import NGram
    noun2predicts = NGram(DATASET).noun2predicts

elif args.model == 'bert':
    from models.lm.mlm_multitok import LM
    prompt2noun2predicts = LM(args.model, DATASET).prompt2noun2predicts

elif args.model == 'roberta':
    from models.lm.mlm_multitok import LM
    prompt2noun2predicts = LM(args.model, DATASET).prompt2noun2predicts

elif args.model == 'gpt2':
    from models.lm.mlm_multitok import LM
    prompt2noun2predicts = LM(args.model, DATASET).prompt2noun2predicts

elif args.model == 'gpt3':
    from models.GPT3.gpt3 import GPT3
    noun2predicts = GPT3(DATASET).noun2predicts
elif args.model == 'vilt':
    from models.ViLT.vilt import ViLT
    prompt2noun2predicts = ViLT(dataset=DATASET).prompt2noun2predicts

elif args.model == 'clip':
    from models.CLIP.clip_openai import CLIP
    noun2predicts = CLIP(DATASET).noun2predicts

elif args.model == 'cem':
    from models.CEM.cem import CEM
    noun2predicts = CEM(DATASET).noun2predicts

    pass
elif args.model == 'cem-pred':
    from models.CEM.cem import CEM
    noun2predicts = CEM(DATASET, predicted_concreteness=True).noun2predicts

if args.model in ['bert', 'roberta', 'vilt', 'gpt2']:
    for prompt, noun2predicts in prompt2noun2predicts.items():
        print(prompt)
        acc_1 = eval.evaluate_acc(noun2predicts, noun2prop, 1, True)
        acc_5 = eval.evaluate_acc(noun2predicts, noun2prop, 5, True)
        r_5 = eval.evaluate_recall(noun2predicts, noun2prop, 5, True)
        r_10 = eval.evaluate_recall(noun2predicts, noun2prop, 10, True)
        mrr = eval.evaluate_rank(noun2predicts, noun2prop, True)
else:
    acc_1 = eval.evaluate_acc(noun2predicts, noun2prop, 1, True)
    acc_5 = eval.evaluate_acc(noun2predicts, noun2prop, 5, True)
    r_5 = eval.evaluate_recall(noun2predicts, noun2prop, 5, True)
    r_10 = eval.evaluate_recall(noun2predicts, noun2prop, 10, True)
    mrr = eval.evaluate_rank(noun2predicts, noun2prop, True)