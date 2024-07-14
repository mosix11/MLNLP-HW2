import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DebertaV2Model, AutoTokenizer, DebertaV2ForSequenceClassification
from transformers.models.deberta_v2.modeling_deberta_v2 import ContextPooler, StableDropout
import os
import sys
import argparse
import datetime
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from src.dataset import AugFEVER
from src.models import DeBERTaNLI
from src.trainers import DefaultTrainer, DefaultEvaluator
from src.utils import nn_utils, utils

from torch.utils.tensorboard import SummaryWriter

cpu = nn_utils.get_cpu_device()
gpu = nn_utils.get_gpu_device()


model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-deberta-v3-base')
tokenizer = AutoTokenizer.from_pretrained('cross-encoder/nli-deberta-v3-base', use_fast=False)



# model.to(gpu)
# model.eval()
# num_corrects = 0
# tot_samples = 0
# with torch.no_grad():
#     for i, batch in enumerate(dl):
#         batch = {k: v.to(gpu) for k, v in batch.items()}
#         scores = model(batch['input_ids'],
#             token_type_ids=batch['token_type_ids'],
#             attention_mask=batch['attention_mask']).logits
#         preds = torch.argmax(scores, dim=1)
#         for j, item in enumerate(batch['labels'].detach().cpu().numpy()):
#             pred = preds[j].detach().cpu().numpy()
#             if item == 0 and pred == 0:
#                 num_corrects += 1
#             elif item == 1 and pred == 2:
#                 num_corrects += 1
#             elif item == 2 and pred == 1:
#                 num_corrects +=1
                
#             tot_samples +=1
# print(num_corrects/tot_samples)





def plot_class_distribution(class_distribution):
    classes = list(class_distribution['train'].keys())
    splits = ['train', 'validation', 'test']
    colors = ['blue', 'orange', 'green']

    x = np.arange(len(classes))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, split in enumerate(splits):
        total = sum(class_distribution[split].values())
        probabilities = [class_distribution[split][cls] / total for cls in classes]
        ax.bar(x + i * width, probabilities, width, label=split.capitalize(), color=colors[i])

    ax.set_xlabel('Class')
    ax.set_ylabel('Probability')
    # ax.set_title('Class Probability Distribution Across Dataset Splits')
    ax.set_xticks(x + width)
    ax.set_xticklabels(classes)
    ax.legend()

    return fig

def plot_train_class_distribution(class_distribution):
    """
    Plot the class probability distribution for the train set.

    Parameters:
    - class_distribution: Dictionary containing class distribution stats for each dataset split.

    Returns:
    - fig: Matplotlib figure object.
    """
    classes = list(class_distribution['train'].keys())
    train_counts = list(class_distribution['train'].values())
    total = sum(train_counts)
    probabilities = [count / total for count in train_counts]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(classes, probabilities, color='green')

    ax.set_xlabel('Class')
    ax.set_ylabel('Probability')
    # ax.set_title('Class Probability Distribution for Train Set')

    return fig

if __name__ == "__main__":
    dataset = AugFEVER(tokenizer,
                       load_augmented=True,
                       use_adv_test=False,
                       batch_size=24,
                       num_workers=4,
                       seed=11)
    # Retrieve class distribution
    print(dataset.get_class_distribution())
    dl = dataset.get_test_dataloader()
    class_distribution = dataset.get_class_distribution()
    
    # deberta = DebertaV2Model.from_pretrained('microsoft/deberta-v3-small', output_hidden_states=True, output_attentions=True)
    # tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-small', use_fast=False)
    # print(deberta.config)
    
    
    # Create the plot
    # fig = plot_class_distribution(class_distribution)
    # fig.savefig('outputs/stats.png', dpi=300, bbox_inches='tight')
    fig = plot_train_class_distribution(class_distribution)
    fig.savefig('outputs/stats.png', dpi=300, bbox_inches='tight')
    # Create a SummaryWriter
    # writer = SummaryWriter(Path('./outputs/tensorboard'))
    
    # Log the plot to TensorBoard
    # utils.plot_to_tensorboard(writer, fig, 'ClassDistribution', 1)
    
    # Close the writer
    # writer.close()
    


