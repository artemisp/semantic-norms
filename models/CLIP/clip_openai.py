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

# DATASET = "concept_properties" # "feature_norms", "memory_colors"
device = "cuda" if th.cuda.is_available() else "cpu"
clip_model_name = "ViT-L/14"
# clip_model_name = "ViT-B/32"
model, preprocess = clip.load(clip_model_name, device=device)
# IMAGE_PATH = f"data/datasets/{DATASET}/images/bing_images/"
# EMBED_PATH = f"data/datasets/{DATASET}/images/image_embeddings/bing_embedding_l14/"


def get_text_embeddings(sentences):
    with th.no_grad():
        text = clip.tokenize(sentences).to(device)
        encoded_text = model.encode_text(text)
    return encoded_text

def get_image_features(noun, image_files, n, EMBED_PATH):
    image_features = []
    for image_file in image_files[noun]:        
        if len(image_features) == n:
            break
        image_id = image_file.split('_')[1].split('.')[0]
        if True:
            image_features.append(pickle.load(open(EMBED_PATH + noun + "_" + image_id + ".p", "rb")))
        # except:
        #     continue
    return image_features

def sort_images(noun, image_files, EMBED_PATH):
    image_features = get_image_features(noun, image_files, len(image_files), EMBED_PATH)
    text_features = get_text_embeddings("A photo of {}.".format(noun))
    image_features = th.vstack(image_features).to(device)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = 100.0 * text_features @ image_features
    indices = th.argsort(similarity[0], descending=True)
    predicts = remove_ties([(image_files[index], float(similarity[0][index])) for index in indices])
    return [pred[0] for pred in predicts]

def remove_ties(predictions):
    new_predictions = [predictions[0]]
    for pred, score in predictions[1:]:
        if abs(score - new_predictions[-1][1]) > 0.001:
            new_predictions.append((pred, score))
    return new_predictions

def get_prediction(n_of_images, prompt, resort=True, DATASET='concept_properties', test=None):
    IMAGE_PATH = f"data/datasets/{DATASET}/images/bing_images/"
    EMBED_PATH = f"data/datasets/{DATASET}/images/image_embeddings/bing_embedding_l14/"
    if DATASET == 'concept_properties' and test:
        noun2prop = pickle.load(open(f"data/datasets/{DATASET}/noun2property/noun2prop_test.p", "rb"))
    else:
        noun2prop = pickle.load(open(f"data/datasets/{DATASET}/noun2property/noun2prop.p", "rb"))    
    candidate_adjs = []
    for noun, props in noun2prop.items():
        candidate_adjs += props
    candidate_adjs = list(set(candidate_adjs))
    all_nouns = list(noun2prop.keys())

    sentences = [prompt.format(adj) for adj in candidate_adjs]
    text_features = get_text_embeddings(sentences)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    clip_predicts = {}
    clip_scores = {}
    noun2sorted_images = {}
    for noun in tqdm(all_nouns):
        if resort:
            image_files = os.listdir(IMAGE_PATH + noun)
            sorted_image_files = sort_images(noun, image_files, EMBED_PATH)
        else:
            sorted_image_files = pickle.load(open(f'data/datasets/{DATASET}/images/noun2sorted_images.p', 'rb'))
        noun2sorted_images[noun] = sorted_image_files
        image_features = get_image_features(noun, sorted_image_files, n_of_images, EMBED_PATH=EMBED_PATH)

        with th.no_grad():
            noun_features = model.encode_text(clip.tokenize("A photo of {}".format(noun)).to(device))
            noun_features /= noun_features.norm(dim=-1, keepdim=True)
        
        if n_of_images == 0:
            similarity = th.mean(noun_features @ text_features.T, dim=0)
        else:
            image_features = th.vstack(image_features).to(device)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            # from pdb import set_trace; set_trace()
            similarity = th.mean(image_features @ text_features.T, dim=0)

        distance = th.ones(similarity.shape[0]) - similarity.cpu()
        indices =  th.argsort(distance, descending=False)
        predicts = [(candidate_adjs[index], float(distance[index])) for index in indices]
        predicts.sort(key=lambda x: x[0])
        predicts.sort(key=lambda x: x[1], reverse=False)
        clip_scores[noun] = predicts
        clip_predicts[noun] = [pred[0] for pred in predicts]
    # pickle.dump(noun2sorted_images, open("../../data/noun2sorted_images.p", "wb"))
    return clip_predicts, clip_scores


class CLIP():
    def __init__(self, dataset, resort=False):
        noun2prop = pickle.load(open(f"data/datasets/{dataset}/noun2property/noun2prop.p", "rb"))
        prompt = "An object with the property of {}."
        self.noun2predicts, scores = get_prediction(n_of_images = 10, prompt=prompt, resort=resort, DATASET=dataset)