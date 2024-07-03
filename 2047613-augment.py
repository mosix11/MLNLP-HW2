import torch
import transformers

import os
import sys
import argparse
import datetime
from pathlib import Path

import datasets



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("mode", help="Specify whether to train the model or test it. Options [\'train\', \'test\']", type=str, default='train')
    # parser.add_argument("-d", "--data", help="Which dataset to use train or test the model. Options [\'original\', \'adversarial\']", type=str, default='original')
    args = parser.parse_args()
    
    dataset = datasets.load_dataset("tommasobonomo/sem_augmented_fever_nli")
    train_set = dataset['train']
    
    
    