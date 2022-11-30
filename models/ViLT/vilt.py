import sys
sys.path.insert(1, '../..')
import eval
from transformers import ViltProcessor, ViltForMaskedLM, ViltFeatureExtractor, BertTokenizer
import os
from PIL import Image
import pickle
from tqdm import tqdm
import torch
import numpy as np

DATASET = "concept_properties" # "feature_norms", "memory_colors"

IMAGE_PATH = f"data/datasets/{DATASET}/images/bing_images/"
EMBED_PATH = f"data/datasets/{DATASET}/images/image_embeddings/vilt_embedding/"
MASK = '[MASK]'

class ViLTScorer():
    def __init__(self):
        with torch.no_grad():
            self.LM = ViltForMaskedLM.from_pretrained("dandelin/vilt-b32-mlm").to("cuda:0")
            self.LM.eval()
        self.image_encoder = ViltFeatureExtractor.from_pretrained("dandelin/vilt-b32-mlm")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")

    def get_token_length(self, text):
        return len(self.tokenizer(text)["input_ids"]) - 2
    
    def encode(self, image_path):
        image = Image.open(image_path)
        return self.image_encoder(image)

    def get_multi_mask_sentence(self, orig_sentence, mask_sentence):
        inputs = self.tokenizer(mask_sentence, return_tensors="pt")["input_ids"]
        labels = self.tokenizer(orig_sentence, return_tensors="pt")["input_ids"]
        if inputs[0].shape[0] != labels[0].shape[0]:
            gap = abs(inputs[0].shape[0] - labels[0].shape[0])
            mask_sentence = mask_sentence.replace(MASK, " ".join([MASK] * (gap + 1)))
        return mask_sentence
    
    def score_masked_batch(self, orig_sentences, mask_sentences, image_feature):
        # inputs = self.processor([image] * len(mask_sentences), mask_sentences, return_tensors="pt", padding=True).to("cuda:0")

        inputs = self.tokenizer(mask_sentences, return_tensors="pt", padding=True).to("cuda:0")
        labels = self.tokenizer(orig_sentences, return_tensors="pt", padding=True)["input_ids"].to("cuda:0")
        mask_token_pos = [[i for i, token_id in enumerate(inputs.data["input_ids"][k]) if token_id == self.tokenizer.mask_token_id] for k in range(inputs.data["input_ids"].shape[0])]
        
        inputs['pixel_values'] = tile(image_feature['pixel_values'][0][None, :, :, :], 0, len(mask_sentences)).to("cuda:0")
        inputs['pixel_mask'] =  tile(image_feature['pixel_mask'][0][None, :, :], 0, len(mask_sentences)).to("cuda:0")

        outputs = self.LM(**inputs)
        logits = outputs.logits.detach()
        scores = []
        for k in range(labels.shape[0]):
            current_logits = logits[k]
            current_labels = labels[k]
            current_mask_token_pos = mask_token_pos[k]
            current_scores = []
            for tok_pos in current_mask_token_pos:
                current_scores.append(float(current_logits[tok_pos][current_labels[tok_pos]]))
            scores.append(np.mean(current_scores))
        return scores
    
def get_prompts(prompt_type, DATASET=DATASET):
	noun2sent = {}
	with open(f"data/datasets/{DATASET}/queries/" + prompt_type + ".prop", "r") as f:
		for raw_data in f.readlines():
			noun = raw_data.split(" :: ")[0]
			sent = raw_data.split(" :: ")[1][:-1]
			noun2sent[noun] = sent
	return noun2sent

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)

class ViLT():
    def __init__(self, dataset, prompt_type='plural',test=True):
        MLM = ViLTScorer()
        IMAGE_PATH = f"data/datasets/{dataset}/images/bing_images/"
        EMBED_PATH = f"data/datasets/{dataset}/images/image_embeddings/vilt_embedding/"
        batch_size = 64
        if dataset == 'concept_properties' and test:
             noun2prop = pickle.load(open(f"data/datasets/{dataset}/noun2property/noun2prop_test.p", "rb"))
        else:
            noun2prop = pickle.load(open(f"data/datasets/{dataset}/noun2property/noun2prop.p", "rb"))
        noun2sorted_images = pickle.load(open(f"data/datasets/{dataset}/images/noun2sorted_images.p", "rb")) 
        candidate_adjs = []
        for noun, props in noun2prop.items():
            candidate_adjs += props
        candidate_adjs = list(set(candidate_adjs))
        n_of_images = 10
        
        noun2image_features = {}
        for noun,_ in noun2prop.items():
            image_files = noun2sorted_images[noun]
            if n_of_images == 0:
                imarray = np.random.rand(224,224,3) * 255
                image = Image.fromarray(imarray.astype('uint8')).convert('RGB')
                image_features = [MLM.image_encoder(image, return_tensors="pt")]
            else:
                image_features = []
                for image_file in image_files:
                    if len(image_features) == n_of_images:
                        break
                    if True:
                        image_id = image_file.split('_')[1].split('.')[0]
                        image_features.append(pickle.load(open(EMBED_PATH + noun + "_" + image_id + ".p", "rb")))
                    
                    # except:
                    #     continue
            noun2image_features[noun] = image_features
        
        if not prompt_type:
            prompt_types = ["plural_most", "singular_can_be", "singular_usually", "singular_generally", "singular", "plural_all", "plural_can_be", "plural_generally", "plural_some", "plural_usually", "plural"]
            prompt_types = prompt_types[6:]
        else:
            prompt_types = [prompt_type]
        prompt2noun2predicts = {}
        for pt in prompt_types:
            print("Getting results for prompt: " + pt)
            noun2sent = get_prompts(prompt_type = pt, DATASET=dataset)
            noun2predicts = {noun: [] for noun in noun2prop}
            for noun, prop in tqdm(noun2prop.items()):
                sent = noun2sent[noun]
                noun2predicts[noun] = []
                pairs = [(sent.replace('[MASK]', adj),sent.replace('[MASK]', MASK)) for adj in candidate_adjs]
                mask_sentences = [MLM.get_multi_mask_sentence(pair[0], pair[1]) for pair in pairs]
                orig_sentences = [pair[0] for pair in pairs]
                iterations = len(mask_sentences) // batch_size
                # image_files = noun2sorted_images[noun]
                image_features = noun2image_features[noun]
                
                # if n_of_images == 0:
                #     imarray = np.random.rand(224,224,3) * 255
                #     image = Image.fromarray(imarray.astype('uint8')).convert('RGB')
                #     image_features = [MLM.image_encoder(image, return_tensors="pt")]
                # else:
                #     image_features = []
                #     for image_file in image_files:
                #         if len(image_features) == n_of_images:
                #             break
                #         try:
                #             image_id = image_file.split('_')[1].split('.')[0]
                #             image_features.append(pickle.load(open(EMBED_PATH + noun + "_" + image_id + ".p", "rb")))
                #         except:
                #             continue
                
                all_image_scores = []
                for image_feature in image_features:
                    scores = []
                    for it in range(iterations + 1):
                        orig_sentences_batch = orig_sentences[it * batch_size : min((it + 1) * batch_size, len(mask_sentences))]
                        mask_sentences_batch = mask_sentences[it * batch_size : min((it + 1) * batch_size, len(mask_sentences))]
                        try:
                            current_scores = MLM.score_masked_batch(orig_sentences_batch, mask_sentences_batch, image_feature)
                            scores += current_scores
                        except:
                            continue
                    all_image_scores.append(scores)
                averaged_scores = np.mean(np.array(all_image_scores), axis=0)
                predicts = [(candidate_adjs[ind], float(scores[ind])) for ind in np.argsort(scores)[::-1]]
                predicts.sort(key=lambda x: x[0])
                predicts.sort(key=lambda x: x[1], reverse=True)
                noun2predicts[noun] = [pred[0] for pred in predicts]
            prompt2noun2predicts[pt] = noun2predicts
        self.prompt2noun2predicts = prompt2noun2predicts


# if __name__ == "__main__":
#     MLM = ViLTScorer()
#     batch_size = 64
#     noun2prop = pickle.load(open(f"../../data/datasets/{DATASET}/noun2property/noun2prop.p", "rb"))
#     noun2sorted_images = pickle.load(open(f"../../data/datasets/{DATASET}/images/noun2sorted_images.p", "rb")) 
#     candidate_adjs = []
#     for noun, props in noun2prop.items():
#         candidate_adjs += props
#     candidate_adjs = list(set(candidate_adjs))

#     n_of_images = 10

#     prompt_types = ["plural_most", "singular_can_be", "singular_usually", "singular_generally", "singular", "plural_all", "plural_can_be", "plural_generally", "plural_some", "plural_usually", "plural"]
#     for pt in prompt_types[6:]:
#         noun2sent = get_prompts(prompt_type = pt)
#         noun2predicts = {noun: [] for noun in noun2prop}
#         for noun, sent in tqdm(noun2sent.items()):
#             noun2predicts[noun] = []
#             pairs = [(sent.replace('[MASK]', adj),sent.replace('[MASK]', MASK)) for adj in candidate_adjs]
#             mask_sentences = [MLM.get_multi_mask_sentence(pair[0], pair[1]) for pair in pairs]
#             orig_sentences = [pair[0] for pair in pairs]
#             iterations = len(mask_sentences) // batch_size

#             image_files = noun2sorted_images[noun]
#             if n_of_images == 0:
#                 imarray = np.random.rand(224,224,3) * 255
#                 image = Image.fromarray(imarray.astype('uint8')).convert('RGB')
#                 image_features = [MLM.image_encoder(image, return_tensors="pt")]
#             else:
#                 image_features = []
#                 for image_file in image_files:
#                     if len(image_features) == n_of_images:
#                         break
#                     try:
#                         image_id = image_file.split('_')[1].split('.')[0]
#                         image_features.append(pickle.load(open(EMBED_PATH + noun + "_" + image_id + ".p", "rb")))
#                     except:
#                         continue

#             all_image_scores = []
#             for image_feature in image_features:
#                 scores = []
#                 for it in range(iterations + 1):
#                     orig_sentences_batch = orig_sentences[it * batch_size : min((it + 1) * batch_size, len(mask_sentences))]
#                     mask_sentences_batch = mask_sentences[it * batch_size : min((it + 1) * batch_size, len(mask_sentences))]
#                     current_scores = MLM.score_masked_batch(orig_sentences_batch, mask_sentences_batch, image_feature)
#                     scores += current_scores
#                 all_image_scores.append(scores)
#             averaged_scores = np.mean(np.array(all_image_scores), axis=0)
#             predicts = [(candidate_adjs[ind], float(scores[ind])) for ind in np.argsort(scores)[::-1]]
#             predicts.sort(key=lambda x: x[0])
#             predicts.sort(key=lambda x: x[1], reverse=True)
#             noun2predicts[noun] = [pred[0] for pred in predicts]

#         for k in [1]:
#             eval.evaluate_acc(noun2predicts, noun2prop, k, True)

#         for k in [1, 5, 10]:
#             eval.evaluate_precision(noun2predicts, noun2prop, k, True)

#         for k in [1, 5, 10]:
#             eval.evaluate_recall(noun2predicts, noun2prop, k, True)