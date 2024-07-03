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
        
        self.metrics = Metrics(self.num_classes)
        
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
        
    def get_tokenizer(self):
        return self.tokenizer
        
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
        self.metrics.update(Y_hat, labels)
        return self.loss(Y_hat, labels)
    
    def loss(self, Y_hat, Y):
        return F.cross_entropy(Y_hat, Y)
    
    def get_metrics(self):
        return {
            'accuracy': self.metrics.accuracy(),
            'precision': self.metrics.precision(),
            'recall': self.metrics.recall(),
            'f1_score': self.metrics.f1_score()
        }
    
    def reset_metrics(self):
        self.metrics = Metrics(self.num_classes)
    # def accuracy(self, preds, labels):
    #     return torch.mean((torch.argmax(preds, dim=1) == labels).to(torch.float64))

    # def precision(self, preds, labels):
    #     num_classes = self.num_classes
    #     _, preds_max = torch.max(preds, 1)
    #     true_positives = torch.zeros(num_classes)
    #     predicted_positives = torch.zeros(num_classes)
        
    #     for i in range(num_classes):
    #         true_positives[i] = ((preds_max == i) & (labels == i)).sum().item()
    #         predicted_positives[i] = (preds_max == i).sum().item()
        
    #     precision_per_class = true_positives / (predicted_positives + 1e-10)
    #     return precision_per_class.mean().item()

    # def recall(self, preds, labels):
    #     num_classes = self.num_classes
    #     _, preds_max = torch.max(preds, 1)
    #     true_positives = torch.zeros(num_classes)
    #     actual_positives = torch.zeros(num_classes)
        
    #     for i in range(num_classes):
    #         true_positives[i] = ((preds_max == i) & (labels == i)).sum().item()
    #         actual_positives[i] = (labels == i).sum().item()
        
    #     recall_per_class = true_positives / (actual_positives + 1e-10)
    #     return recall_per_class.mean().item()

    # def f1_score(self, preds, labels,):
    #     num_classes = self.num_classes
    #     prec = self.precision(preds, labels, num_classes)
    #     rec = self.recall(preds, labels, num_classes)
        
    #     f1_per_class = 2 * (prec * rec) / (prec + rec + 1e-10)
    #     return f1_per_class
    
class Metrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0
        self.true_positives = torch.zeros(self.num_classes)
        self.predicted_positives = torch.zeros(self.num_classes)
        self.actual_positives = torch.zeros(self.num_classes)

    def update(self, preds, labels):
        _, preds_max = torch.max(preds, 1)
        self.correct += (preds_max == labels).sum().item()
        self.total += labels.size(0)

        for i in range(self.num_classes):
            self.true_positives[i] += ((preds_max == i) & (labels == i)).sum().item()
            self.predicted_positives[i] += (preds_max == i).sum().item()
            self.actual_positives[i] += (labels == i).sum().item()

    def accuracy(self):
        return self.correct / self.total

    def precision(self):
        precision_per_class = self.true_positives / (self.predicted_positives + 1e-10)
        return precision_per_class.mean().item()

    def recall(self):
        recall_per_class = self.true_positives / (self.actual_positives + 1e-10)
        return recall_per_class.mean().item()

    def f1_score(self):
        prec = self.precision()
        rec = self.recall()
        return 2 * (prec * rec) / (prec + rec + 1e-10)