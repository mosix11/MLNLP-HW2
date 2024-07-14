import torch

import datasets
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding
from collections import Counter

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
        if load_augmented and not os.listdir(root_dir):
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
        self.reverse_label_map = {0: 'CONTRADICTION', 1: 'NEUTRAL', 2: 'ENTAILMENT'}
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
    
    def decode_tokens(self, token_ids):
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        # decoded_text = self.tokenizer.convert_tokens_to_string(tokens)
        return tokens
    
    def prepare_samples(self, samples):
        pass # TODO implement this function for test time! it should get a sample or a batch of samples and return the tokenized form of the inputs
             # ready for inputing to the model and get the predictions.
        
    def _init_loaders(self):
        if self.load_augmented:
            
            dataset_train_aug = datasets.Dataset.load_from_disk(self.root_dir)
            dataset_orig = datasets.load_dataset("tommasobonomo/sem_augmented_fever_nli")
            # print(dataset_train_aug.features, '\n\n')
            # print(dataset_orig['train'].features, '\n\n')
            train_set = datasets.concatenate_datasets([dataset_train_aug.remove_columns(['wsd', 'srl']), dataset_orig['train'].remove_columns(['wsd', 'srl'])])
            # train_set = dataset_train_aug.remove_columns(['wsd', 'srl'])
            val_set = dataset_orig['validation'].remove_columns(['wsd', 'srl'])
            test_set = dataset_orig['test'].remove_columns(['wsd', 'srl'])
        else:
            dataset = datasets.load_dataset("tommasobonomo/sem_augmented_fever_nli")
            train_set = dataset['train'].remove_columns(['wsd', 'srl'])
            val_set = dataset['validation'].remove_columns(['wsd', 'srl'])
            test_set = dataset['test'].remove_columns(['wsd', 'srl'])
            
        if self.use_adv_test:
            adv_testset = datasets.load_dataset("iperbole/adversarial_fever_nli")
            test_set = adv_testset['test']
        
       

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
            
            
    def get_class_distribution(self):
        splits = {
            'train': Counter(),
            'validation': Counter(),
            'test': Counter()
        }
        for batch in self.get_train_dataloader():
            labels = batch['labels'].numpy()
            splits['train'].update([self.reverse_label_map[label] for label in labels])
        
        for batch in self.get_val_dataloader():
            labels = batch['labels'].numpy()
            splits['validation'].update([self.reverse_label_map[label] for label in labels])
            
        for batch in self.get_test_dataloader():
            labels = batch['labels'].numpy()
            splits['test'].update([self.reverse_label_map[label] for label in labels])
            
        splits['train'] = dict(splits['train'])
        splits['validation'] = dict(splits['validation'])
        splits['test'] = dict(splits['test'])
        return splits
            
        