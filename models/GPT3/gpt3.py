import eval
import pickle

DATASET = "concept_properties" # "feature_norms", "memory_colors"

class GPT3():
    def __init__(self, dataset):
        if dataset == 'memory_color':
            self.noun2predicts = {line.split(',')[0]: line.split(',')[1].strip().split() for line in open(f'data/datasets/{dataset}/GPT3/gpt3_predicts.txt', 'r').readlines()}
        else:
            self.noun2predicts = {line.split()[0]: line.split()[1:] for line in open(f'data/datasets/{dataset}/GPT3/gpt3_predicts.txt', 'r').readlines()}
