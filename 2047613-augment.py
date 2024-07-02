import torch
import transformers

import os
import sys
import argparse
import datetime
from pathlib import Path

from src.dataset import AugFEVER



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="Specify whether to train the model or test it. Options [\'train\', \'test\']", type=str, default='train')
    parser.add_argument("-d", "--data", help="Which dataset to use train or test the model. Options [\'original\', \'adversarial\']", type=str, default='original')
    args = parser.parse_args()
    
    
    if args.mode == 'train':
        if args.data == 'original':
            pass
        elif args.data == 'adversarial':
            pass
        else:
            raise RuntimeError('Invalid dataset type! Possible options are [\'original\', \'adversarial\']')
    
    elif args.mode == 'test':
        if args.data == 'original':
            pass
        elif args.data == 'adversarial':
            pass
        else:
            raise RuntimeError('Invalid dataset type! Possible options are [\'original\', \'adversarial\']')

    else:
        raise RuntimeError('Invalid mode! Possible options are [\'train\', \'test\']')
    