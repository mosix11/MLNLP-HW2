import torch

from ..utils import nn_utils
class HFDatasetNLI():
    
    def __init__(self, dataset, tokenizer) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        
        self.label_map = {'CONTRADICTION': 0, 'NEUTRAL': 1, 'ENTAILMENT': 2}
        self.reverse_label_map = {0: 'CONTRADICTION', 1: 'NEUTRAL', 2: 'ENTAILMENT'}
        
        self.cpu = nn_utils.get_cpu_device()  
        
    def get_tokenizer(self):
        return self.tokenizer
    
    def tokenize_function(self, samples):
        labels = samples['label']
        tokenized_inputs = self.tokenizer(samples["premise"], samples["hypothesis"], truncation=True, max_length=512, return_tensors="pt")
        tokenized_inputs['labels'] = torch.tensor([self.label_map[label] for label in labels] if isinstance(labels, list) else self.label_map[labels])
        return tokenized_inputs
    
    
    def getitem_tokenized(self, idx, device=None):
        if device == None: device = self.cpu
        tokenized_sample = self.tokenize_function(self.dataset[idx])
        tokenized_sample = {k: v.to(device) for k, v in tokenized_sample.items()}
        return tokenized_sample
    
    def tokenize_sample(self, sample, device=None):
        if device == None: device = self.cpu
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
    
    def __len__(self):
        return len(self.dataset)