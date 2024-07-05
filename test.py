import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import sys
import argparse
import datetime
import time
from pathlib import Path

from src.dataset import AugFEVER
from src.models import DeBERTaNLI
from src.trainers import DefaultTrainer, DefaultEvaluator
from src.utils import nn_utils

cpu = nn_utils.get_cpu_device()
gpu = nn_utils.get_gpu_device()


model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-deberta-v3-base')
tokenizer = AutoTokenizer.from_pretrained('cross-encoder/nli-deberta-v3-base', use_fast=False)

dataset = AugFEVER(tokenizer,
                       load_augmented=False,
                       use_adv_test=True,
                       batch_size=24,
                       num_workers=4,
                       seed=11)

dl = dataset.get_test_dataloader()
model.to(gpu)
model.eval()
num_corrects = 0
tot_samples = 0
with torch.no_grad():
    for i, batch in enumerate(dl):
        batch = {k: v.to(gpu) for k, v in batch.items()}
        scores = model(batch['input_ids'],
            token_type_ids=batch['token_type_ids'],
            attention_mask=batch['attention_mask']).logits
        preds = torch.argmax(scores, dim=1)
        for j, item in enumerate(batch['labels'].detach().cpu().numpy()):
            pred = preds[j].detach().cpu().numpy()
            if item == 0 and pred == 0:
                num_corrects += 1
            elif item == 1 and pred == 2:
                num_corrects += 1
            elif item == 2 and pred == 1:
                num_corrects +=1
                
            tot_samples +=1
print(num_corrects/tot_samples)