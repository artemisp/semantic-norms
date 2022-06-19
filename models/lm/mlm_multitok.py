import sys
sys.path.insert(1, '../..')
import eval
import os
import numpy as np
import pickle
import copy
import torch
from tqdm import tqdm
from nltk.corpus import wordnet as wn
from transformers import BertTokenizer, BertForMaskedLM
from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers import GPT2Tokenizer, GPT2LMHeadModel

cache_dir = os.environ['TRANSFORMERS_CACHE']

DATASET = "concept_properties" # "feature_norms", "memory_colors"

model_name = "roberta-large"
# model_name = "bert-large-uncased"
# model_name = "gpt2-large"



class MLMScorer():
	"""A LM scorer for the conditional probability of an ending given a prompt."""

	def __init__(self, model_name):
		self.model_name = model_name
		if 'roberta' in model_name:
			self.MASK = '<mask>'
		elif 'bert' in model_name:
			self.MASK = '[MASK]'
		elif 'gpt' in model_name:
			self.MASK = 'mask'
		

		if "roberta" in model_name:
			self.model_type = "roberta"
			with torch.no_grad():
				self.LM = RobertaForMaskedLM.from_pretrained(model_name, cache_dir=cache_dir).to("cuda:0")
				self.LM.eval()
			self.tokenizer = RobertaTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

		elif "bert" in model_name:
			self.model_type = "bert"
			with torch.no_grad():
				self.LM = BertForMaskedLM.from_pretrained(model_name, cache_dir=cache_dir).to("cuda:0")
				self.LM.eval()
			self.tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

		elif "gpt" in model_name:
			self.model_type = "gpt"
			with torch.no_grad():
				self.LM = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=cache_dir).to("cuda:0")
				self.LM.eval()
			self.tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
		else:
			raise ValueError(f"Unknown model name: {model_name}.")

	def get_token_length(self, text):
		return len(self.tokenizer(text)["input_ids"]) - 2
	
	def get_multi_mask_sentence(self, orig_sentence, mask_sentence):
		inputs = self.tokenizer(mask_sentence, return_tensors="pt")["input_ids"]
		labels = self.tokenizer(orig_sentence, return_tensors="pt")["input_ids"]
		if inputs[0].shape[0] != labels[0].shape[0]:
			gap = abs(inputs[0].shape[0] - labels[0].shape[0])
			mask_sentence = mask_sentence.replace(self.MASK, " ".join([self.MASK] * (gap + 1)))
		return mask_sentence

	def score_single(self, sentence):
		# score the ppl of the entire single sentence
		orig_inputs = self.tokenizer(sentence, return_tensors="pt").to("cuda:0")
		labels = orig_inputs.data["input_ids"]
		n_tokens = len(labels)

		total_loss = 0
		for i in range(n_tokens):
			inputs = copy.deepcopy(orig_inputs)
			inputs["input_ids"][:, i] = self.tokenizer.mask_token_id

			with torch.no_grad():
				outputs = self.LM(**inputs, labels=labels)
				loss = outputs.loss.detach().item()
			total_loss += loss

		return total_loss/len(labels)

	def get_token_id(self, text, token):
		tokens = self.tokenizer.tokenize(text)
		target = None
		if self.model_type == "roberta":
			target = "Ġ" + token
		elif self.model_type == "gpt":
			target = "Ġ" + token
		elif self.model_type == "bert":
			target = token
		return tokens.index(target), tokens

	def score_masked(self, orig_sentence, mask_sentence):
		# score the ppl of the masked tokens in mask_sentence being the corresponding tokens in orig_sentence
		inputs = self.tokenizer(mask_sentence, return_tensors="pt").to("cuda:0")
		labels = self.tokenizer(orig_sentence, return_tensors="pt")["input_ids"].to("cuda:0")
		
		if inputs["input_ids"][0].shape[0] != labels[0].shape[0]:
			gap = abs(inputs["input_ids"][0].shape[0] - labels[0].shape[0])
			mask_sentence = mask_sentence.replace(self.MASK, " ".join([self.MASK] * (gap + 1)))
			inputs = self.tokenizer(mask_sentence, return_tensors="pt").to("cuda:0")

		if 'gpt' in self.model_name:
			mask_token_pos = [i for i, token_id in enumerate(inputs.data["input_ids"][0]) if token_id == 9335]
		else:
			mask_token_pos = [i for i, token_id in enumerate(inputs.data["input_ids"][0]) if token_id == self.tokenizer.mask_token_id]

		for pos in range(len(labels[0])):
			if pos not in mask_token_pos:
				labels[:, pos] = -100

		outputs = self.LM(**inputs, labels=labels)
		loss = outputs.loss.detach().item()
		return np.exp(loss)
	
	def score_masked_batch(self, orig_sentences, mask_sentences):
		if 'gpt' in self.model_name:
			self.tokenizer.pad_token = self.tokenizer.eos_token

		inputs = self.tokenizer(mask_sentences, return_tensors="pt", padding=True).to("cuda:0")
		labels = self.tokenizer(orig_sentences, return_tensors="pt", padding=True)["input_ids"].to("cuda:0")
		
		if 'gpt' in self.model_name:
			mask_token_pos = [[i for i, token_id in enumerate(inputs.data["input_ids"][k]) if token_id == 9335] for k in range(inputs.data["input_ids"].shape[0])]
		else:
			mask_token_pos = [[i for i, token_id in enumerate(inputs.data["input_ids"][k]) if token_id == self.tokenizer.mask_token_id] for k in range(inputs.data["input_ids"].shape[0])]
		
		outputs = self.LM(**inputs, labels=labels)
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

def get_prompts(prompt_type, dataset=DATASET):
	noun2sent = {}
	with open(f"../data/datasets/{dataset}/queries/" + prompt_type + ".prop", "r") as f:
		for raw_data in f.readlines():
			noun = raw_data.split(" :: ")[0]
			sent = raw_data.split(" :: ")[1][:-1]
			noun2sent[noun] = sent
	return noun2sent

class LM():
	def __init__(self, model_name, dataset):
		if model_name == 'bert':
			model_name = "bert-large-uncased"
		elif model_name == 'roberta':
			model_name = "roberta-large"
		elif model_name == 'gpt2':
			model_name = "gpt2-large"

		LM = MLMScorer(model_name)
		batch_size = 64
		noun2prop = pickle.load(open(f"../data/datasets/{dataset}/noun2property/noun2prop.p", "rb"))
		candidate_adjs = []
		for noun, props in noun2prop.items():
			candidate_adjs += props
		candidate_adjs = list(set(candidate_adjs))
		
		prompt_types = ["plural_most", "singular_can_be", "singular_usually", "singular_generally", "singular", "plural_all", "plural_can_be", "plural_generally", "plural_some", "plural_usually", "plural"]
		prompt2noun2predicts = {}
		for pt in prompt_types:
			print("Getting results for prompt: " + pt)
			noun2sent = get_prompts(prompt_type = pt, dataset=dataset)
			noun2predicts = {}
			noun2scores = {}
			for noun, sent in tqdm(noun2sent.items()):
				noun2predicts[noun] = []
				pairs = [(sent.replace('[MASK]', adj),sent.replace('[MASK]', self.MASK)) for adj in candidate_adjs]
				if 'gpt' in model_name:
					scores = [LM.score_masked(pair[0], pair[1]) for pair in pairs]
					predicts = [(candidate_adjs[ind], float(scores[ind])) for ind in np.argsort(scores)]
					predicts.sort(key=lambda x: x[0])
					predicts.sort(key=lambda x: x[1])
				else:
					mask_sentences = [LM.get_multi_mask_sentence(pair[0], pair[1]) for pair in pairs]
					orig_sentences = [pair[0] for pair in pairs]
					iterations = len(mask_sentences) // batch_size
					scores = []
					for it in range(iterations + 1):
						orig_sentences_batch = orig_sentences[it * batch_size : min((it + 1) * batch_size, len(mask_sentences))]
						mask_sentences_batch = mask_sentences[it * batch_size : min((it + 1) * batch_size, len(mask_sentences))]
						current_scores = LM.score_masked_batch(orig_sentences_batch, mask_sentences_batch)
						scores += current_scores
					predicts = [(candidate_adjs[ind], float(scores[ind])) for ind in np.argsort(scores)[::-1]]
					predicts.sort(key=lambda x: x[0])
					predicts.sort(key=lambda x: x[1], reverse=True)
					noun2predicts[noun] = [pred[0] for pred in predicts]
			prompt2noun2predicts[pt] = noun2predicts
		self.prompt2noun2predicts = prompt2noun2predicts


# if __name__ == "__main__":
# 	LM = MLMScorer(model_name)
# 	batch_size = 64
# 	noun2prop = pickle.load(open(f"../data/datasets/{DATASET}/noun2property/noun2prop.p", "rb"))
# 	candidate_adjs = []
# 	for noun, props in noun2prop.items():
# 		candidate_adjs += props
# 	candidate_adjs = list(set(candidate_adjs))
	
# 	prompt_types = ["plural_most", "singular_can_be", "singular_usually", "singular_generally", "singular", "plural_all", "plural_can_be", "plural_generally", "plural_some", "plural_usually", "plural"]
# 	for pt in prompt_types:
# 		noun2sent = get_prompts(prompt_type = pt)
# 		noun2predicts = {}
# 		noun2scores = {}
# 		for noun, sent in tqdm(noun2sent.items()):
# 			noun2predicts[noun] = []
# 			pairs = [(sent.replace('[MASK]', adj),sent.replace('[MASK]', MASK)) for adj in candidate_adjs]
# 			if 'gpt' in model_name:
# 				scores = [LM.score_masked(pair[0], pair[1]) for pair in pairs]
# 				predicts = [(candidate_adjs[ind], float(scores[ind])) for ind in np.argsort(scores)]
# 				predicts.sort(key=lambda x: x[0])
# 				predicts.sort(key=lambda x: x[1])
# 			else:
# 				mask_sentences = [LM.get_multi_mask_sentence(pair[0], pair[1]) for pair in pairs]
# 				orig_sentences = [pair[0] for pair in pairs]
# 				iterations = len(mask_sentences) // batch_size
# 				scores = []
# 				for it in range(iterations + 1):
# 					orig_sentences_batch = orig_sentences[it * batch_size : min((it + 1) * batch_size, len(mask_sentences))]
# 					mask_sentences_batch = mask_sentences[it * batch_size : min((it + 1) * batch_size, len(mask_sentences))]
# 					current_scores = LM.score_masked_batch(orig_sentences_batch, mask_sentences_batch)
# 					scores += current_scores
# 				predicts = [(candidate_adjs[ind], float(scores[ind])) for ind in np.argsort(scores)[::-1]]
# 				predicts.sort(key=lambda x: x[0])
# 				predicts.sort(key=lambda x: x[1], reverse=True)
# 			noun2predicts[noun] = [pred[0] for pred in predicts]
# 		print(pt)
# 		pickle.dump(noun2predicts, open("/nlp/data/yueyang/prototypicality/clean/output/output_{}/{}+{}.p".format(DATASET, model_name, pt), "wb"))