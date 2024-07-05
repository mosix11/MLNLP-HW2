import torch
import transformers
import ollama
import datasets
from sentence_transformers import CrossEncoder
import os
import sys
import argparse
import datetime
from pathlib import Path
import copy
from src.utils import nn_utils

cpu = nn_utils.get_cpu_device()
gpu = nn_utils.get_gpu_device()

weights_dir = Path('./weights').absolute()
nli_model_weights = weights_dir.joinpath('base_nli_model.pt')

class HFDatasetNLI():
    
    def __init__(self, dataset, tokenizer) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        
        self.label_map = {'CONTRADICTION': 0, 'NEUTRAL': 1, 'ENTAILMENT': 2}
        self.reverse_label_map = {0: 'CONTRADICTION', 1: 'NEUTRAL', 2: 'ENTAILMENT'}        
        
    def tokenize_function(self, samples):
        labels = samples['label']
        tokenized_inputs = self.tokenizer(samples["premise"], samples["hypothesis"], truncation=True, max_length=512, return_tensors="pt")
        tokenized_inputs['labels'] = torch.tensor([self.label_map[label] for label in labels] if isinstance(labels, list) else self.label_map[labels])
        return tokenized_inputs
    
    
    def getitem_tokenized(self, idx, device=cpu):
        tokenized_sample = self.tokenize_function(self.dataset[idx])
        tokenized_sample = {k: v.to(device) for k, v in tokenized_sample.items()}
        return tokenized_sample
    
    def tokenize_sample(self, sample, device=cpu):
        tokenized_sample = self.tokenize_function(sample)
        tokenized_sample = {k: v.to(device) for k, v in tokenized_sample.items()}
        return tokenized_sample
    
    def decode_tokens(self, token_ids):
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        # decoded_text = self.tokenizer.convert_tokens_to_string(tokens)
        return tokens
    
    def get_label(self, X):
        return self.reverse_label_map[X]
    
    def __getitem__(self, idx):
        return self.dataset[idx]

ollama_client = ollama.Client(host='http://localhost:11434')
def prompt_ollama(prmpt):
    # ollama.chat(model='llama3', messages=[{'role': 'user', 'content': 'Why is the sky blue?'}])
    resp = ollama_client.generate(model='llama3', prompt=prmpt)
    return resp
    
    
def paraphrase(text):
    prompt = """Paraphraser the following text and make sure the meaning does not change at all and make sure you just give me the paraphrased sentence without any extra word. ```text: {}```""".format(text)
    response = prompt_ollama(prompt)
    return response['response']
    
def clean_print(data_sample):
    print('\nPremise: {}\nHypothesis: {}\nLabel: {}\n'.format(data_sample['premise'], data_sample['hypothesis'], data_sample['label']))
    

def get_nli_prediction(model, sample):
    pred, att_scores = model.predict(sample, return_attention_scores=True)
    pred = pred.detach().cpu().item()
    att_scores = [las.detach().cpu() for las in att_scores]
    return pred, att_scores
        

def get_top_k_tokens_contributing_to_cls(model, sample, tokenizer, num_layers=4, k=4):
    # Forward pass to get the attention scores
    pred, att_scores = get_nli_prediction(model, sample)
    
    # Extract the token ids and attention scores
    input_ids = sample["input_ids"].detach().cpu().squeeze()
    token_type_ids = sample["token_type_ids"].detach().cpu().squeeze()
    
    # Identify punctuation tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    punctuation_tokens = ["!", ".", ",", "?", ":", ";", "-", "_", "(", ")", "[", "]", "{", "}", "'", "\"", "/", "▁."]
    punctuation_mask = torch.tensor([t in punctuation_tokens or (t.startswith("▁") and t[1:] in punctuation_tokens) for t in tokens])
    
    # Use attention scores from the last num_layers layers
    last_layers_attention = torch.stack(att_scores[-num_layers:]).mean(dim=0).squeeze()  # Shape: (heads, seq_len, seq_len)
    
    # Average attention scores across all heads in the selected layers
    attention_scores_mean = last_layers_attention.mean(axis=0)  # Shape: (seq_len, seq_len)
    
    # Identify the position of the [CLS] token
    cls_index = (input_ids == tokenizer.cls_token_id).nonzero(as_tuple=True)[0]
    
    # Separate premise and hypothesis using token_type_ids
    premise_mask = (token_type_ids == 0)
    hypothesis_mask = (token_type_ids == 1)
    
    
    # Calculate the attention scores contributing to [CLS] token
    cls_attention_scores = attention_scores_mean[:, cls_index].squeeze()
    
    # Exclude [CLS], [SEP], and punctuation tokens from the final ranking
    exclude_tokens_mask = (input_ids != tokenizer.cls_token_id) & (input_ids != tokenizer.sep_token_id) & (~punctuation_mask)
    premise_exclude_mask = premise_mask & exclude_tokens_mask
    hypothesis_exclude_mask = hypothesis_mask & exclude_tokens_mask

    # Indices for ranking (excluding [CLS], [SEP], and punctuation)
    premise_exclude_indices = premise_exclude_mask.nonzero(as_tuple=True)[0]
    hypothesis_exclude_indices = hypothesis_exclude_mask.nonzero(as_tuple=True)[0]

    # Filter the scores for ranking
    filtered_premise_scores = cls_attention_scores[premise_exclude_indices]
    filtered_hypothesis_scores = cls_attention_scores[hypothesis_exclude_indices]

    # Identify the top-k important tokens, ensuring k is not greater than the number of tokens
    k_premise = min(k, len(filtered_premise_scores))
    k_hypothesis = min(k, len(filtered_hypothesis_scores))
    
    top_k_premise_indices = filtered_premise_scores.argsort(descending=True)[:k_premise]
    top_k_hypothesis_indices = filtered_hypothesis_scores.argsort(descending=True)[:k_hypothesis]
    
    # Get the top-k token IDs for premise and hypothesis
    top_k_premise_token_ids = input_ids[premise_exclude_indices[top_k_premise_indices]].tolist()
    top_k_hypothesis_token_ids = input_ids[hypothesis_exclude_indices[top_k_hypothesis_indices]].tolist()
    
    return top_k_premise_token_ids, top_k_hypothesis_token_ids


        
def get_top_k_important_tokens_from_nli_model(model, sample, tokenizer, num_layers=4, k=4):
    # Forward pass to get the attention scores
    pred, att_scores = get_nli_prediction(model, sample)
    
    # Extract the token ids and attention scores
    input_ids = sample["input_ids"].detach().cpu().squeeze()
    token_type_ids = sample["token_type_ids"].detach().cpu().squeeze()
    
    # Identify punctuation tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    punctuation_tokens = ["!", ".", ",", "?", ":", ";", "-", "_", "(", ")", "[", "]", "{", "}", "'", "\"", "/", "▁."]
    punctuation_mask = torch.tensor([t in punctuation_tokens or (t.startswith("▁") and t[1:] in punctuation_tokens) for t in tokens])
    
    # Use attention scores from the last num_layers layers
    last_layers_attention = torch.stack(att_scores[-num_layers:]).mean(dim=0).squeeze()  # Shape: (heads, seq_len, seq_len)
    
    # Average attention scores across all heads in the selected layers
    attention_scores_mean = last_layers_attention.mean(axis=0)  # Shape: (seq_len, seq_len)
    
    # Separate premise and hypothesis using token_type_ids
    premise_mask = (token_type_ids == 0)
    hypothesis_mask = (token_type_ids == 1)
    
    premise_indices = premise_mask.nonzero(as_tuple=True)[0]
    hypothesis_indices = hypothesis_mask.nonzero(as_tuple=True)[0]
    
    # Calculate cross attention scores
    premise_to_hypothesis_scores = attention_scores_mean[premise_indices][:, hypothesis_indices].sum(axis=1)
    hypothesis_to_premise_scores = attention_scores_mean[hypothesis_indices][:, premise_indices].sum(axis=1)
    
    # Exclude [CLS], [SEP], and punctuation tokens from the final ranking
    exclude_tokens_mask = (input_ids != tokenizer.cls_token_id) & (input_ids != tokenizer.sep_token_id) & (~punctuation_mask)
    premise_exclude_mask = premise_mask & exclude_tokens_mask
    hypothesis_exclude_mask = hypothesis_mask & exclude_tokens_mask

    # Indices for ranking (excluding [CLS], [SEP], and punctuation)
    premise_exclude_indices = premise_exclude_mask.nonzero(as_tuple=True)[0]
    hypothesis_exclude_indices = hypothesis_exclude_mask.nonzero(as_tuple=True)[0]

    # Filter the scores for ranking
    filtered_premise_scores = premise_to_hypothesis_scores[premise_exclude_indices - premise_indices[0]]
    filtered_hypothesis_scores = hypothesis_to_premise_scores[hypothesis_exclude_indices - hypothesis_indices[0]]

    # Identify the top-k important tokens, ensuring k is not greater than the number of tokens
    k_premise = min(k, len(filtered_premise_scores))
    k_hypothesis = min(k, len(filtered_hypothesis_scores))
    
    top_k_premise_indices = filtered_premise_scores.argsort(descending=True)[:k_premise]
    top_k_hypothesis_indices = filtered_hypothesis_scores.argsort(descending=True)[:k_hypothesis]
    
    # Get the top-k token IDs for premise and hypothesis
    top_k_premise_token_ids = input_ids[premise_exclude_indices[top_k_premise_indices]].tolist()
    top_k_hypothesis_token_ids = input_ids[hypothesis_exclude_indices[top_k_hypothesis_indices]].tolist()
    
    return top_k_premise_token_ids, top_k_hypothesis_token_ids




def get_sentences_similarity(model, sentence1, sentence2):
    if isinstance(sentence1, list) and isinstance(sentence2, list):
        inputs = zip(sentence1, sentence2)
    else:
        inputs = [(sentence1, sentence2)]
    scores = model.predict(inputs)
    print(scores)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("mode", help="Specify whether to train the model or test it. Options [\'train\', \'test\']", type=str, default='train')
    # parser.add_argument("-d", "--data", help="Which dataset to use train or test the model. Options [\'original\', \'adversarial\']", type=str, default='original')
    args = parser.parse_args()
    
    full_dataset = datasets.load_dataset("tommasobonomo/sem_augmented_fever_nli")
    train_set = full_dataset['train']
    nli_model = torch.load(nli_model_weights).to(gpu)
    stsb_model = CrossEncoder("cross-encoder/stsb-roberta-base") # cross-encoder/stsb-roberta-large
    stsb_model.model.to(gpu)
    
    # sentence1 = "Coding is so hard."
    # sentence2 = paraphrase(sentence1)
    # get_sentences_similarity(stsb_model, 'Coding is so hard.', 'Writing computer code is extremely easy.')
    dataset = HFDatasetNLI(train_set, nli_model.get_tokenizer())
    
    # sample_idx = 0
    # orig_sample = dataset[sample_idx]
    # aug_sample = copy.deepcopy(orig_sample)
    
    # orig_sample_t = dataset.tokenize_sample(orig_sample, device=gpu)
    # aug_sample['hypothesis'] = paraphrase(aug_sample['hypothesis'])
    # aug_sample_t = dataset.tokenize_sample(orig_sample, device=gpu)
    
    # pred_o, _ = get_nli_prediction(nli_model, orig_sample_t)
    # pred_a, _ = get_nli_prediction(nli_model, aug_sample_t)
    
    # clean_print(orig_sample)
    # print("Predicted label for original sample: ", pred_o)
    # clean_print(aug_sample)
    # print("Predicted label for augmented sample: ", pred_a)
    # sample_tokenized = dataset.getitem_tokenized(sample_idx, device=gpu)
    
    print(dataset[2000])
    get_nli_prediction(nli_model, dataset.getitem_tokenized(1, device=gpu))
    clean_print(dataset[2000])
    prem_tokens, hyp_tokens = get_top_k_important_tokens_from_nli_model(nli_model, dataset.getitem_tokenized(2000, device=gpu), dataset.tokenizer, num_layers=4, k=4)
    print(dataset.decode_tokens(prem_tokens))
    print(dataset.decode_tokens(hyp_tokens))
    
    prem_tokens, hyp_tokens = get_top_k_tokens_contributing_to_cls(nli_model, dataset.getitem_tokenized(2000, device=gpu), dataset.tokenizer, num_layers=4, k=4)
    print(dataset.decode_tokens(prem_tokens))
    print(dataset.decode_tokens(hyp_tokens))
    
    # clean_print(train_set[0])
    
    
    
    