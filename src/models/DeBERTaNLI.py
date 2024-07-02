import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import DebertaV2Model, AutoTokenizer, DebertaV2ForSequenceClassification
from transformers.models.deberta_v2.modeling_deberta_v2 import ContextPooler, StableDropout



class DeBERTaNLI(nn.Module):
    
    def __init__(self,
                 checkpoint:str = "microsoft/deberta-v3-small",
                 num_classes:int = 3,
                 dropout:float = None) -> None:
        super().__init__()
        
        self.deberta = DebertaV2Model.from_pretrained(checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
        
        self.num_classes = num_classes
        self.pooler = ContextPooler(self.deberta.config)
        output_dim = self.pooler.output_dim
        
        self.classifier = nn.Linear(output_dim, self.num_classes)
        
        drop_out = self.deberta.config.hidden_dropout_prob if dropout is None else dropout
        self.dropout = StableDropout(drop_out)
        
        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=0.1),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.1),
        #     nn.Linear(hidden_size, num_labels)
        # )
        
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        
        transformer_out = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        
        encoded_seq = transformer_out[0]
        encoded_seq = self.pooler(encoded_seq)
        encoded_seq = self.dropout(encoded_seq)
        
        logits = self.classifier(encoded_seq)
        return logits
        
        
    def training_step(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        labels = batch["labels"]
        
        Y_hat = self(input_ids, attention_mask, token_type_ids)
        return self.loss(Y_hat, labels)
    
    def validation_step(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        labels = batch["labels"]
        
        Y_hat = self(input_ids, attention_mask, token_type_ids)
        return self.loss(Y_hat, labels), self.accuracy(Y_hat, labels)
    
    def loss(self, Y_hat, Y):
        return F.cross_entropy(Y_hat, Y)
    
    def accuracy(self, predictions, targets):
        return torch.mean((predictions == targets).to(torch.float64))
        
    def get_tokenizer(self):
        return self.tokenizer