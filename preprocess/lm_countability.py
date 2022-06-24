import pickle
import numpy as np
from tqdm import tqdm
import os
from transformers import pipeline

def main():
    model_name = 'bert-large-uncased'
    unmasker = pipeline('fill-mask', model=model_name, device = 0)
    noun2prop = pickle.load(open("../data/datasets/concept_properties/noun2property/noun2prop.p", "rb"))
    noun2det = {}
    for noun in tqdm(noun2prop):
        outputs = unmasker("I have [MASK] {}.".format(noun), targets = ["a", "an"])
        order = [output['token_str'] for output in outputs]
        noun2det[noun] = order[0]
    pickle.dump(noun2det, open("../inter_data/noun2det_CSLB.p", "wb"))

if __name__ == "__main__":
    main()