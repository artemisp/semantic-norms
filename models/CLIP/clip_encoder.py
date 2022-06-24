import sys
sys.path.insert(1, '../..')
import eval
import pickle
from tqdm import tqdm
import torch as th
import numpy as np
import clip
import os
import random
from PIL import Image

DATASET = "concept_properties" # "feature_norms", "memory_colors"

device = "cuda" if th.cuda.is_available() else "cpu"
clip_model_name = "ViT-L/14"
model, preprocess = clip.load(clip_model_name, device=device)
IMAGE_PATH = f"data/datasets/{DATASET}/images/bing_images/"
EMBED_PATH = f"data/datasets/{DATASET}/images/image_embeddings/bing_embedding_l14/"

def get_text_embeddings(sentences):
    with th.no_grad():
        text = clip.tokenize(sentences).to(device)
        return model.encode_text(text)

if __name__ == "__main__":
    noun2prop = pickle.load(open(f"data/datasets/{DATASET}/noun2property/noun2prop.p", "rb"))
    all_nouns = list(noun2prop.keys())
    clip_predicts = {}
    for noun in tqdm(all_nouns):
        image_files = os.listdir(IMAGE_PATH + noun)
        with th.no_grad():
            for image_file in image_files:
                print(image_file)
                image_id = image_file.split('_')[1].split('.')[0]
                try:
                    image_input = preprocess(Image.open(IMAGE_PATH + noun + "/" + image_file)).unsqueeze(0).to(device)
                    image_feature = model.encode_image(image_input).to(device)
                    pickle.dump(image_feature.cpu(), open(EMBED_PATH + noun + "_" + image_id + ".p", "wb"))
                except:
                    print(noun, image_file)