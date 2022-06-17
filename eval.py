import pickle
import sys
import numpy as np
word2concretness = pickle.load(open("/nlp/data/yueyang/prototypicality/MRD/data/word2concreteness.M.p", "rb"))

def evaluate_recall(noun2predicts, noun2prop, K, PRINT):
    results = []
    nouns = list(noun2prop.keys())
    for noun in nouns:
        predicts = noun2predicts[noun]
        correct = 0
        for pred in predicts[:K]:
            if pred in noun2prop[noun]:
                correct += 1
        results.append(correct / len(noun2prop[noun]))
    if PRINT:
        print("recall@{}: ".format(K), np.mean(results))
    return np.mean(results)

def evaluate_precision(noun2predicts, noun2prop, K, PRINT):
    results = []
    nouns = list(noun2prop.keys())
    for noun in nouns:
        predicts = noun2predicts[noun]
        correct = 0
        for pred in predicts[:K]:
            if pred in noun2prop[noun]:
                correct += 1
        results.append(correct / K)
    if PRINT:
        print("precision@{}: ".format(K), np.mean(results))
    return np.mean(results)

def evaluate_acc(noun2predicts, noun2prop, K, PRINT):
    num_of_correct = 0
    nouns = list(noun2prop.keys())
    for noun in nouns:
        predicts = noun2predicts[noun]
        correct = False
        for pred in predicts[:K]:
            if pred in noun2prop[noun]:
                correct = True
                break
        if correct == True:
            num_of_correct += 1
    if PRINT:
        print("top{} acc: ".format(K), num_of_correct / len(noun2prop))
    return num_of_correct / len(noun2prop)

def evaluate_rank(noun2predicts, noun2prop, PRINT):
    nouns = list(noun2prop.keys())
    rrs = []
    rs = []
    results = []
    for noun in nouns:
        predicts = noun2predicts[noun]
        for prop in noun2prop[noun]:
            results.append((prop, predicts.index(prop) + 1))
            rrs.append(1 / (predicts.index(prop) + 1))
            rs.append(predicts.index(prop) + 1)
    if PRINT:
        print("MRR: {}\nMedian rank: {}\nMean rank: {}\n".format(np.mean(rrs), np.mean(rs), np.median(rs)))
    return results

if __name__ == "__main__":
    predict_file = sys.argv[1]
    noun2predicts = pickle.load(open(predict_file, "rb"))
    noun2prop = pickle.load(open("/nlp/data/yueyang/prototypicality/MRD/data/MRD/MRD_noun2prop.p", "rb"))
    evaluate_acc(noun2predicts, noun2prop)