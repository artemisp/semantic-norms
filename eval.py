import pickle
import sys
import numpy as np

def evaluate_recall(noun2predicts, noun2prop, K, PRINT):
    results = []
    nouns = list(noun2prop.keys())
    for noun in nouns:
        predicts = noun2predicts[noun]
        correct = 0
        if len(noun2prop[noun]) == 0:
            continue
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
        print("MRR: {}\nMean rank: {}\nMedian rank: {}\n".format(np.mean(rrs), np.mean(rs), np.median(rs)))
    return results