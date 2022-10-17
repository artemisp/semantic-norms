from transformers import ViltFeatureExtractor
import os
from PIL import Image
import torch
import numpy as np
import pickle
from tqdm import tqdm
import time
import sys
sys.path.append('../..')

DATASET = "concept_properties" # "concept_properties", "feature_norms", "memory_colors"

IMAGE_PATH = "/nlp/data/yueyang/prototypicality/semantic-norms-main/" + f"data/datasets/{DATASET}/images/bing_images/"
EMBED_PATH =  "/nlp/data/yueyang/prototypicality/semantic-norms-main/" + f"data/datasets/{DATASET}/images/image_embeddings/vilt_embedding/"

if __name__ == "__main__":
    noun2prop = pickle.load(open(f"data/datasets/{DATASET}/noun2property/noun2prop.p", "rb"))
    all_nouns = list(noun2prop.keys())
    image_encoder = ViltFeatureExtractor.from_pretrained("dandelin/vilt-b32-mlm")
    start = time.time()
    for noun in tqdm(all_nouns):
        image_files = os.listdir(IMAGE_PATH + noun)
        for image_file in tqdm(image_files):
            image_id = image_file.split('_')[1].split('.')[0]
            if noun + "_" + image_id + ".p" in os.listdir(EMBED_PATH):
                    continue
            try:
                image = Image.open(IMAGE_PATH + noun + "/" + image_file)
                image_feature = image_encoder(image.convert('RGB'), return_tensors="pt")
                pickle.dump(image_feature, open(EMBED_PATH + noun + "_" + image_id + ".p", "wb"))
            except:
                print(noun, image_id)
    print(f"features computed in {time.time()-start} seconds")