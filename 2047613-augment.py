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
from src.dataset import HFDatasetNLI
from src.utils import nn_utils
import string
import re
import random
import nltk
from nltk.corpus import wordnet as wn
from tqdm import tqdm

cpu = nn_utils.get_cpu_device()
gpu = nn_utils.get_gpu_device()

weights_dir = Path('./weights').absolute()
nli_model_weights = weights_dir.joinpath('model.pt')



ollama_client = ollama.Client(host='http://localhost:11434')
def prompt_ollama(prmpt):
    # ollama.chat(model='llama3', messages=[{'role': 'user', 'content': 'Why is the sky blue?'}])
    resp = ollama_client.generate(model='llama3', prompt=prmpt)
    return resp
    
    
def paraphrase(text):
    prompt = """Paraphrase the following text and make sure the meaning does not change at all. Return only the paraphrase. You must not include any explanations or comments.
    Original text: ```{}```
    Paraphrased text: """.format(text)
    
    response = prompt_ollama(prompt)
    return response['response']
    

def replace_with_synonym_llama(text, word):
    prompt = """Replace the word `{}` in the following text with its synonym. Return only the modified text. You must not include any explanations or comments.
    Original text: ```{}```
    Modified text: """.format(word, text)
    response = prompt_ollama(prompt)
    return response['response']

def replace_with_antonym_llama(text, word):
    prompt = """Replace the word `{}` in the following text with its antonym. Return only the modified text. You must not include any explanations or comments.
    Original text: ```{}```
    Modified text: """.format(word, text)
    response = prompt_ollama(prompt)
    return response['response']

def replace_with_hypernym_llama(text, word):
    prompt = """Replace the word `{}` in the following text with its hypernym. Return only the modified text. You must not include any explanations or comments.
    Original text: ```{}```
    Modified text: """.format(word, text)
    response = prompt_ollama(prompt)
    # if response['response'] == '-1':
    #     print('**************  No hypernym *************')
    #     response = paraphrase(text)
    return response['response']

def nli_using_llama(premise, hypothesis):
    prompt = """Consider the following NLI task. Given the premise and hypothesis only return the correct label. You must not include any explanations or comments.
    Premise: ```{}```
    Hypothesis: ```{}```
    Labels: ```[CONTRADICTION, NEUTRAL, ENTAILMENT]```""".format(premise, hypothesis)
    response = prompt_ollama(prompt)
    
    return response['response'].strip(string.punctuation)

def get_nli_prediction(model, sample):
    pred, att_scores = model.predict(sample)
    pred = pred.detach().cpu().item()
    att_scores = [las.detach().cpu() for las in att_scores]
    return pred, att_scores
        

def get_top_k_tokens_contributing_to_cls(model, sample, tokenizer, num_layers=1, k=3):
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
    
    # Get the top-k token IDs and positions for premise and hypothesis
    top_k_premise = [(input_ids[idx].item(), idx.item()) for idx in premise_exclude_indices[top_k_premise_indices]]
    top_k_hypothesis = [(input_ids[idx].item(), idx.item()) for idx in hypothesis_exclude_indices[top_k_hypothesis_indices]]
    
    return top_k_premise, top_k_hypothesis


        
def get_top_k_important_tokens_from_nli_model(model, sample, tokenizer, num_layers=1, k=3):
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
    
    # Get the top-k token IDs and positions for premise and hypothesis
    top_k_premise = [(input_ids[idx].item(), idx.item()) for idx in premise_exclude_indices[top_k_premise_indices]]
    top_k_hypothesis = [(input_ids[idx].item(), idx.item()) for idx in hypothesis_exclude_indices[top_k_hypothesis_indices]]
    
    return top_k_premise, top_k_hypothesis


def get_full_word_of_token(tokenizer, sample, top_tokens_p, top_tokens_h):
    # Tokenize the input sentences
    sentence1 = sample['premise']
    sentence2 = sample['hypothesis']
    
    inputs = tokenizer(sentence1, sentence2, return_tensors='pt', truncation=True, max_length=512)
    input_ids = inputs['input_ids'][0]
    token_type_ids = inputs['token_type_ids'][0]
    
    # Decode the input_ids to get the tokens
    decoded_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Initialize the lists for storing full words and their positions
    full_words_p = []
    full_words_h = []

    # Function to clean token (remove special characters at the start)
    def clean_token(token):
        return token.lstrip('Ġ▁')

    # Function to strip punctuation from the start and end of a word
    def strip_punctuation(word):
        return word.strip(string.punctuation)

    # Special tokens to ignore
    special_tokens = {'[CLS]', '[SEP]', '[PAD]', '[MASK]'}

    # Helper function to process tokens and extract full words with positions
    def extract_full_words(top_tokens, token_type, sentence):
        full_words = []
        for token_id, position in top_tokens:
            if position >= len(decoded_tokens) or decoded_tokens[position] in special_tokens or token_type_ids[position] != token_type:
                continue

            # Initialize current word with the cleaned token at the given position
            current_word = clean_token(decoded_tokens[position])
            pos = position

            # Check preceding tokens to complete the word
            while pos > 0 and decoded_tokens[pos - 1] not in special_tokens and not decoded_tokens[pos].startswith(('Ġ', '▁')) and token_type_ids[pos - 1] == token_type:
                pos -= 1
                current_word = clean_token(decoded_tokens[pos]) + current_word

            # Check following tokens to complete the word
            pos = position
            while pos + 1 < len(decoded_tokens) and decoded_tokens[pos + 1] not in special_tokens and not decoded_tokens[pos + 1].startswith(('Ġ', '▁')) and token_type_ids[pos + 1] == token_type:
                pos += 1
                current_word += clean_token(decoded_tokens[pos])

            # Strip punctuation from the full word
            full_word = strip_punctuation(current_word)

            # Find the starting position of the full word in the original sentence
            sentence_lower = sentence.lower()
            full_word_lower = full_word.lower()

            # Find the start position in the respective sentence
            start_pos = sentence_lower.find(full_word_lower)

            # Add the full word and its starting position to the result list
            full_words.append((full_word, start_pos))
        return full_words

    # Extract full words for top_tokens_p (sentence1)
    full_words_p = extract_full_words(top_tokens_p, 0, sentence1)

    # Extract full words for top_tokens_h (sentence2)
    full_words_h = extract_full_words(top_tokens_h, 1, sentence2)

    return full_words_p, full_words_h




def find_wsd_srl_for_word(sample, word_pos, p_h='p'):
    word, position = word_pos
    query = 'premise' if p_h == 'p' else 'hypothesis'
    wsd = []
    srl = []
    
    
    for item in sample['wsd'][query]:
        if item["text"] == word:
            wsd.append(item)
    
    for token in sample['srl'][query]['tokens']:
        if token["rawText"] == word:
            tmp_tk = copy.deepcopy(token)
            for annot in sample['srl'][query]['annotations']:
                if tmp_tk['index'] == annot['tokenIndex']:
                    tmp_tk['annotation'] = annot
            srl.append(tmp_tk)
            
    find_all = lambda w, s: [m.start() for m in re.finditer(rf'\b{re.escape(w)}\b', s)]
    if len(wsd) > 1:
        idxs = find_all(word, sample[query])
        for i, idx in enumerate(idxs):
            if idx == position:
                wsd = [wsd[i]]
                break
    if len(srl) > 1:
        pass
    return wsd, srl
    

def get_sentences_similarity(model, sentence1, sentence2):
    if isinstance(sentence1, list) and isinstance(sentence2, list):
        inputs = zip(sentence1, sentence2)
    else:
        inputs = [(sentence1, sentence2)]
    scores = model.predict(inputs)
    return scores[0]

def get_detailed_synset_info(nltk_synset_id, original_word):
    try:
        # Retrieve synset using the ID
        synset = wn.synset(nltk_synset_id)
        
        # Get synonyms, excluding the original word
        synonyms = [lemma.name() for lemma in synset.lemmas() if lemma.name() != original_word]
        
        # Get antonyms
        antonyms = []
        for lemma in synset.lemmas():
            if lemma.antonyms():
                antonyms.extend([antonym.name() for antonym in lemma.antonyms()])
        
        # Get hypernyms
        hypernyms = [hypernym.name() for hypernym in synset.hypernyms()]
        
        # Get hyponyms
        hyponyms = [hyponym.name() for hyponym in synset.hyponyms()]
        
        # Get other relations (holonyms and meronyms as examples)
        holonyms = [holonym.name() for holonym in synset.member_holonyms()]
        meronyms = [meronym.name() for meronym in synset.part_meronyms()]
        
        return {
            "synonyms": synonyms,
            "antonyms": antonyms,
            "hypernyms": hypernyms,
            "hyponyms": hyponyms,
            "holonyms": holonyms,
            "meronyms": meronyms
        }
    except Exception as e:
        return {"error": str(e)}
    
def augment_sample(nli_model, stsb_model, sample):
    tokenizer = nli_model.get_tokenizer()
    prem_tokens, hyp_tokens = get_top_k_tokens_contributing_to_cls(nli_model, dataset.tokenize_sample(sample, device=gpu), tokenizer, num_layers=4, k=4)
    prem_top_words, hyp_top_words = get_full_word_of_token(tokenizer, sample, prem_tokens, hyp_tokens)
    
    wsd, srl = find_wsd_srl_for_word(sample, hyp_top_words[0], p_h='h')
    
    def check_empty_wsd(wsd):
        if wsd['bnSynsetId'] == "O" and wsd['wnSynsetOffset'] == "O" and wsd['nltkSynset'] == "O":
            return True
        else: return False
    
    def augment_by_hyp(sample, wsd, srl, hyp_word):
        aug_sample = copy.deepcopy(sample)
        # Augument the sample with parahprasing the premise and changing the most important word with its synonym 
        new_prem = paraphrase(sample['premise'])
        similarity_score_prem = get_sentences_similarity(stsb_model, sample['premise'], new_prem)
        cnt = 0
        while similarity_score_prem < 0.75:
            # print('Premise similarity score loop!')
            new_prem = paraphrase(sample['premise'])
            similarity_score_prem = get_sentences_similarity(stsb_model, sample['premise'], new_prem)
            cnt +=1
            if cnt > 2:
                break
        # if wsd == None:
        #     rnd_number = random.random()
        #     if rnd_number < 0.5:
        #         new_hyp = replace_with_synonym_llama(sample['hypothesis'], hyp_word[0])
        #         similarity_score_hyp = get_sentences_similarity(stsb_model, sample['hypothesis'], new_hyp)
        #         cnt = 0
        #         while similarity_score_hyp < 0.8:
        #             # print('Hypothesis synonym similarity score loop!')
        #             new_hyp = replace_with_synonym_llama(sample['hypothesis'], hyp_word[0])
        #             similarity_score_hyp = get_sentences_similarity(stsb_model, sample['hypothesis'], new_hyp)
        #             cnt +=1
        #             if cnt > 0:
        #                 break
        #         aug_sample['premise'] = new_prem
        #         aug_sample['hypothesis'] = new_hyp
        #     else:
        #         # Augument the sample with parahprasing the premise and changing the most important word with its antonym and reversing the label
        #         new_hyp = replace_with_antonym_llama(sample['hypothesis'], hyp_word[0])
        #         similarity_score_hyp = get_sentences_similarity(stsb_model, sample['hypothesis'], new_hyp)
        #         cnt = 0
        #         while similarity_score_hyp > 0.50:
        #             # print('Hypothesis antonym similarity score loop!')
        #             new_hyp = replace_with_antonym_llama(sample['hypothesis'], hyp_word[0])
        #             similarity_score_hyp = get_sentences_similarity(stsb_model, sample['hypothesis'], new_hyp)
        #             cnt +=1
        #             if cnt > 0:
        #                 break
        #         aug_sample['premise'] = new_prem
        #         aug_sample['hypothesis'] = new_hyp
        #         if sample['label'] == 'ENTAILMENT':
        #             aug_sample['label'] = 'CONTRADICTION'
        #         elif sample['label'] == 'CONTRADICTION':
        #             aug_sample['label'] = 'ENTAILMENT'
            
        #     return aug_sample
        # else:
            # nltk_ID = wsd['nltkSynset']
            # word_info = get_detailed_synset_info(nltk_ID, hyp_word[0])
            # print(hyp_word)
            # print(word_info)
        
        rnd_number = random.random()
        if rnd_number < 0.1:
            new_hyp = replace_with_hypernym_llama(sample['hypothesis'], hyp_word[0])
            similarity_score_hyp = get_sentences_similarity(stsb_model, sample['hypothesis'], new_hyp)
            cnt = 0
            while similarity_score_hyp < 0.75:
                # print('Hypothesis hypernym similarity score loop!')
                new_hyp = replace_with_hypernym_llama(sample['hypothesis'], hyp_word[0])
                similarity_score_hyp = get_sentences_similarity(stsb_model, sample['hypothesis'], new_hyp)
                cnt +=1
                if cnt > 0:
                    break
            aug_sample['premise'] = new_prem
            aug_sample['hypothesis'] = new_hyp
        elif rnd_number >= 0.1 and rnd_number<= 0.5:
            new_hyp = replace_with_antonym_llama(sample['hypothesis'], hyp_word[0])
            similarity_score_hyp = get_sentences_similarity(stsb_model, sample['hypothesis'], new_hyp)
            cnt = 0
            while similarity_score_hyp > 0.50:
                # print('Hypothesis antonym similarity score loop!')
                new_hyp = replace_with_antonym_llama(sample['hypothesis'], hyp_word[0])
                similarity_score_hyp = get_sentences_similarity(stsb_model, sample['hypothesis'], new_hyp)
                cnt +=1
                if cnt > 0:
                    break
            aug_sample['premise'] = new_prem
            aug_sample['hypothesis'] = new_hyp
            if sample['label'] == 'ENTAILMENT':
                aug_sample['label'] = 'CONTRADICTION'
            elif sample['label'] == 'CONTRADICTION':
                aug_sample['label'] = 'ENTAILMENT'
        else:
            new_hyp = replace_with_synonym_llama(sample['hypothesis'], hyp_word[0])
            similarity_score_hyp = get_sentences_similarity(stsb_model, sample['hypothesis'], new_hyp)
            cnt = 0
            while similarity_score_hyp < 0.75:
                # print('Hypothesis synonym similarity score loop!')
                new_hyp = replace_with_synonym_llama(sample['hypothesis'], hyp_word[0])
                similarity_score_hyp = get_sentences_similarity(stsb_model, sample['hypothesis'], new_hyp)
                cnt +=1
                if cnt > 0:
                    break
            aug_sample['premise'] = new_prem
            aug_sample['hypothesis'] = new_hyp
        

        return aug_sample
                            
    def augment_by_prem(sample, was, srl, prem_word):
        pass
    
    if len(wsd) == 0:
        augumented_sample = augment_by_hyp(sample, None, None, hyp_top_words[0])
        
    elif len(wsd) == 1:
        if check_empty_wsd(wsd[0]):
            wsd, srl = find_wsd_srl_for_word(sample, hyp_top_words[1], p_h='h')
            if len(wsd) == 0:
                augumented_sample = augment_by_hyp(sample, None, None, hyp_top_words[0])
            elif len(wsd) == 1:
                if check_empty_wsd(wsd[0]):
                    augumented_sample = augment_by_hyp(sample, None, None, hyp_top_words[0])
                else:
                    augumented_sample = augment_by_hyp(sample, wsd[0], srl, hyp_top_words[1])
            else:
                augumented_sample = augment_by_hyp(sample, None, None, hyp_top_words[0]) 
        else:
            augumented_sample = augment_by_hyp(sample, wsd[0], srl, hyp_top_words[0])
    
    elif len(wsd) > 1 :
        # print('ohh nooo')
        wsd, srl = find_wsd_srl_for_word(sample, hyp_top_words[1], p_h='h')
        if len(wsd) == 0:
            augumented_sample = augment_by_hyp(sample, None, None, hyp_top_words[0])
        elif len(wsd) == 1:
            augumented_sample = augment_by_hyp(sample, wsd, srl, hyp_top_words[1])
        elif len(wsd) > 1:
            augumented_sample = augment_by_hyp(sample, None, None, hyp_top_words[0])
            
    return augumented_sample


def clean_print(data_sample):
    print('\nPremise: {}\nHypothesis: {}\nLabel: {}\n'.format(data_sample['premise'], data_sample['hypothesis'], data_sample['label']))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    nltk.download('wordnet')
    full_dataset = datasets.load_dataset("tommasobonomo/sem_augmented_fever_nli")
    train_set = full_dataset['train']
    nli_model = torch.load(nli_model_weights).to(gpu)
    stsb_model = CrossEncoder("cross-encoder/stsb-roberta-base") # cross-encoder/stsb-roberta-large
    stsb_model.model.to(gpu)
    

    dataset = HFDatasetNLI(train_set, nli_model.get_tokenizer())

    augmented_ds = []
    for i, sample in tqdm(enumerate(dataset), total=len(dataset), desc="Aumenting Samples"):
        aug_sample = augment_sample(nli_model, stsb_model, dataset[i])
        augmented_ds.append(aug_sample)

    augmented_dataset = datasets.Dataset.from_list(augmented_ds)
    augmented_dataset.save_to_disk('./data/')


    
    

    