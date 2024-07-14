import torch

import os
import sys
import argparse
import datetime
import time
from pathlib import Path

from src.dataset import AugFEVER
from src.models import DeBERTaNLI
from src.trainers import DefaultTrainer, DefaultEvaluator

torch.backends.cudnn.benchmark = True

def parse_args(args):
    training_mode = False
    use_augmented_trainingset = False
    use_adv_testset = False
    if args.mode == 'train':
        training_mode = True
        if args.data == 'original': pass
        elif args.data == 'adversarial':use_augmented_trainingset = True
        else: raise RuntimeError('Invalid dataset type! Possible options are [\'original\', \'adversarial\']')
    elif args.mode == 'test':
        if args.data == 'original': pass
        elif args.data == 'adversarial': use_adv_testset = True
        else: raise RuntimeError('Invalid dataset type! Possible options are [\'original\', \'adversarial\']')
    else:
        raise RuntimeError('Invalid mode! Possible options are [\'train\', \'test\']')
    return training_mode, use_augmented_trainingset, use_adv_testset

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
    # base_trained_model_weights = weights_dir.joinpath('base_nli_model.pt')
    # aug_trained_model_weights = weights_dir.joinpath('aug_nli_model.pt')
    trained_model_weights = weights_dir.joinpath('model.pt')

    model = DeBERTaNLI(dropout=0.3)
    tokenizer = model.get_tokenizer()
    training_mode, use_augmented_trainingset, use_adv_testset = parse_args(args)
    dataset = AugFEVER(tokenizer,
                       root_dir=root_dir,
                       outputs_dir=outputs_dir,
                       load_augmented=use_augmented_trainingset,
                       use_adv_test=use_adv_testset,
                       batch_size=24,
                       num_workers=4,
                       seed=11)

    if training_mode:
        trainer = DefaultTrainer(max_epochs=3,
                                lr=1e-5,
                                optimizer_type="adamw",
                                run_on_gpu=True,
                                )
        trainer.fit(model, dataset, resume=False)
        
        # if use_augmented_trainingset: torch.save(model, aug_trained_model_weights)
        # else: torch.save(model, base_trained_model_weights)
        torch.save(model, trained_model_weights)
    else:
        # if use_augmented_trainingset:
        #     if not aug_trained_model_weights.exists():
        #         raise RuntimeError('Model weights not found! You should train the model first!')
        #     model = torch.load(aug_trained_model_weights)
        #     # model = torch.load(base_trained_model_weights)
        # else:
        #     if not base_trained_model_weights.exists():
        #         raise RuntimeError('Model weights not found! You should train the model first!')
        #     model = torch.load(base_trained_model_weights)
        #     # model = torch.load(aug_trained_model_weights)
        model = torch.load(trained_model_weights)
        model.eval()
        evaluator = DefaultEvaluator(run_on_gpu=True)
        print(evaluator.evaluate(model, dataset))
        
    
        
    # dl = dataset.get_train_dataloader()
    # for i, batch in enumerate(dl):
    #     print(type(batch["input_ids"]))
