import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from ..utils import nn_utils, utils
import os
import socket
import datetime
from pathlib import Path
import time
from tqdm import tqdm

class DefaultEvaluator():
    
    def __init__(self,
                 run_on_gpu=False
                 ) -> None:
        
        self.cpu = nn_utils.get_cpu_device()
        self.gpu = nn_utils.get_gpu_device()
        
        if self.gpu == None and run_on_gpu:
            raise RuntimeError("""gpu device not found!""")
        self.run_on_gpu = run_on_gpu      
    
    
    
    def prepare_data(self, dataset):
        self.test_dataloader = dataset.get_test_dataloader()
        self.num_test_batches = len(self.test_dataloader)
        
    def prepare_model(self, model):
        if self.run_on_gpu:
            model.to(self.gpu)
        self.model = model
        
    def prepare_batch(self, batch):
        if self.run_on_gpu:
            batch = {k: v.to(self.gpu) for k, v in batch.items()}
        return batch
    

    
    def evaluate(self, model, dataset):
        self.prepare_data(dataset)
        self.prepare_model(model)
        
        self.model.eval()
        self.model.reset_metrics()
        loss = utils.AverageMeter()
        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.test_dataloader), total=self.num_test_batches, desc="Processing Batches"):
                b_loss = self.model.validation_step(self.prepare_batch(batch))
                loss.update(b_loss.detach().cpu().numpy())
        
        return loss.avg, self.model.get_metrics() 
        