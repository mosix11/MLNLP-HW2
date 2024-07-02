import torch

import datasets
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding


import os
import sys
from pathlib import Path


class AugFEVER():
    
    def __init__(self,
                 tokenizer,
                 root_dir:Path = Path('./data').absolute(),
                 outputs_dir:Path = Path('./outputs').absolute(),
                 load_augmented:bool = False,
                 use_adv_test:bool = False,
                 batch_size:int = 64,
                 num_workers:int = 2,
                 seed:int = 11,
                 ) -> None:
        
        if not root_dir.exists():
            raise RuntimeError("The root directory does not exist!")
        if not outputs_dir.exists():
            raise RuntimeError("The output directory does not exist!")
        if load_augmented and not root_dir.joinpath('aug path').exists():
            raise RuntimeError("The augmented dataset does not exist! You should run the augmentation script first!!")
            
                
        self.tokenizer = tokenizer
        self.root_dir = root_dir
        self.outputs_dir = outputs_dir
        self.load_augmented = load_augmented
        self.use_adv_test = use_adv_test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        
        self.label_map = {'CONTRADICTION': 0, 'NEUTRAL': 1, 'ENTAILMENT': 2}
        self._init_loaders()
        
        
    def get_train_dataloader(self):
        return self.train_loader
    
    def get_val_dataloader(self):
        return self.val_loader
    
    def get_test_dataloader(self):
        return self.test_loader
    
    def get_tokenizer(self):
        return self.tokenizer
        
    def tokenize_function(self, samples):
        labels = samples['label']
        tokenized_inputs = self.tokenizer(samples["premise"], samples["hypothesis"], truncation=True, max_length=512)
        tokenized_inputs['labels'] = [self.label_map[label] for label in labels]
        return tokenized_inputs
    
    def prepare_samples(self, samples):
        pass # TODO implement this function for test time! it should get a sample or a batch of samples and retur the tokenized form of the inputs
             # ready for inputing to the model and get the predictions.
        
    def _init_loaders(self):
        if self.load_augmented:
            dataset = datasets.Dataset.load_from_disk("path to augmented ds")
        else:
            dataset = datasets.load_dataset("tommasobonomo/sem_augmented_fever_nli")
        if self.use_adv_test:
            adv_testset = datasets.load_dataset("iperbole/adversarial_fever_nli")
        
        train_set = dataset['train']
        val_set = dataset['validation']
        test_set = dataset['test'] if not self.use_adv_test else adv_testset['test']

        # def check_long_sequences(dataset, max_length=512):
        #     long_sequences = []
        #     for idx, sample in enumerate(dataset):
        #         tokens = self.tokenizer.encode(sample["premise"], sample["hypothesis"])
        #         if len(tokens) > max_length:
        #             long_sequences.append((idx, len(tokens)))
        #     return long_sequences

        # long_sequences = check_long_sequences(train_set)
        # print(f"Number of long sequences: {len(long_sequences)}")
        # for idx, length in long_sequences:
        #     print(f"Sample {idx} has {length} tokens")
        
        self.train_loader = self._build_dataloader(train_set)
        self.val_loader = self._build_dataloader(val_set)
        self.test_loader = self._build_dataloader(test_set)
        
        
    def _build_dataloader(self, dataset):
        dataset = dataset.map(self.tokenize_function, batched=True)
        features = list(dataset.features.keys())
        remove_features = [ftr for ftr in features if not ftr in ['input_ids', 'token_type_ids', 'attention_mask', 'labels']]
        dataset = dataset.remove_columns(remove_features)
        dataset.set_format('torch')
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator, num_workers=self.num_workers, pin_memory=True) 
            