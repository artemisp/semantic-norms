sys.path.insert(1, '../..')
import eval
import pickle

DATASET = "concept_properties" # "feature_norms", "memory_colors"

class GPT3():
    def __init__(self, dataset):
        self.noun2predicts = {line.split()[0]: line.split()[1:] for line in open(f'../../data/datasets/{dataset}/GPT3/gpt3_predicts.txt', 'r').readlines()}



# noun2predicts = {line.split()[0]: line.split()[1:] for line in open(f'../data/datasets/{DATASET}/GPT3/gpt3_predicts.txt', 'r').readlines()}
# noun2prop = pickle.load(open(f"/datasets/{DATASET}/noun2property/noun2prop.p", "rb"))

# eval.evaluate(noun2predicts, noun2prop, PRINT=True)
# eval.evaluate_acc(noun2predicts, noun2prop,  PRINT=True)