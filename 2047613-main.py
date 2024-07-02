import torch

import os
import sys
import argparse
import datetime
import time
from pathlib import Path

from src.dataset import AugFEVER
from src.models import DeBERTaNLI
from src.trainers import DefaultTrainer

def parse_args(args):
    use_augmented_trainingset = False
    use_adv_testset = False
    if args.mode == 'train':
        if args.data == 'original': pass
        elif args.data == 'adversarial':use_augmented_trainingset = True
        else: raise RuntimeError('Invalid dataset type! Possible options are [\'original\', \'adversarial\']')
    elif args.mode == 'test':
        if args.data == 'original': pass
        elif args.data == 'adversarial': use_adv_testset = True
        else: raise RuntimeError('Invalid dataset type! Possible options are [\'original\', \'adversarial\']')
    else:
        raise RuntimeError('Invalid mode! Possible options are [\'train\', \'test\']')
    return use_augmented_trainingset, use_adv_testset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="Specify whether to train the model or test it. Options [\'train\', \'test\']", type=str)
    parser.add_argument("-d", "--data", help="Which dataset to use train or test the model. Options [\'original\', \'adversarial\']", type=str)
    args = parser.parse_args()
    
    root_dir = Path('./data').absolute()
    outputs_dir = Path('./outputs').absolute()
    weights_dir = Path('./weights').absolute()
    if not root_dir.exists(): os.mkdir(root_dir)
    if not outputs_dir.exists(): os.mkdir(outputs_dir)
    if not weights_dir.exists(): os.mkdir(weights_dir)

    model = DeBERTaNLI()
    tokenizer = model.get_tokenizer()
    use_augmented_trainingset, use_adv_testset = parse_args(args)
    dataset = AugFEVER(tokenizer,
                       root_dir=root_dir,
                       outputs_dir=outputs_dir,
                       load_augmented=use_augmented_trainingset,
                       use_adv_test=use_adv_testset,
                       batch_size=32,
                       num_workers=4,
                       seed=11)
    
    trainer = DefaultTrainer(max_epochs=5,
                             lr=1e-5,
                             optimizer_type="adamw",
                             run_on_gpu=True,
                             )
    trainer.fit(model, dataset, resume=False)
    torch.save(model, weights_dir)
    # dl = dataset.get_train_dataloader()
    # for i, batch in enumerate(dl):
    #     print(type(batch["input_ids"]))
