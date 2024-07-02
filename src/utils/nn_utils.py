import torch
import os
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import collections
from . import utils
from torch.utils.data import Dataset
from typing import Any

def get_gpu_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else : return None
    
def get_cpu_device():
    return torch.device('cpu')

class DataModule():

    def __init__(self, root='./data/', num_workers=8) -> None:
        
        if not os.path.exists(root):
            os.mkdir(root)
        self.root = root
        self.num_workers = num_workers

    
    def get_dataloader(self, train=True):
        pass


    def train_dataloader(self):
        return self.get_dataloader(train=True)
    

    def val_dataloader(self):
        return self.get_dataloader(train=False)


class SimpleListDataset(Dataset):
    
    def __init__(self, data_list:list) -> None:
        super().__init__()
        self.data_list = data_list
        
    def __getitem__(self, index) -> Any:
        return self.data_list[index]
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    
def init_constant(module, const):
    if type(module) == nn.Linear:
        nn.init.constant_(module.weight, const)
        nn.init.zeros_(module.bias)
    if type(module) == nn.Conv2d:
        nn.init.constant_(module.weight, const)
        nn.init.zeros_(module.bias)

def init_uniform(module, l=0, u=1):
    if type(module) == nn.Linear:
        nn.init.uniform_(module.weight, a=l, b=u)
        nn.init.zeros_(module.bias)
    if type(module) == nn.Conv2d:
        nn.init.uniform_(module.weight, a=l, b=u)
        nn.init.zeros_(module.bias)

def init_normal(module, mean=0, std=0.01):
    if type(module) == nn.Linear:
        nn.init.normal_(module.weight, mean=mean, std=std)
        nn.init.zeros_(module.bias)
    if type(module) == nn.Conv2d:
        nn.init.normal_(module.weight, mean=mean, std=std)
        nn.init.zeros_(module.bias)

def init_xavier_uniform(module, gain=1):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)
    if type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)
    if type(module) == nn.GRU:
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(module._parameters[param])

def init_xavier_normal(module, gain=1):
    if type(module) == nn.Linear:
        nn.init.xavier_normal_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)
    if type(module) == nn.Conv2d:
        nn.init.xavier_normal_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)
    if type(module) == nn.GRU:
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_normal_(module._parameters[param])
    

def lazy_layer_initialization(model, dummy_input, init_method=None):
    model(*dummy_input)
    if init_method is not None:
        model.apply(init_method)


def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis."""
    # X: 3D tensor, valid_lens: 1D or 2D tensor 
    # 1D valid_lens specifies valid length for each batch However the 2d valid_lens specifies valid lengths also 
    # for each sample in each batch aswell meaning samples in the same batch can have different lengths.
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X

    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)




def calculate_ouput_dim(net, input_dim=None):
    
    batch_size = 8
    for child in net.children():
        if isinstance(child, nn.Sequential):
            child = child[0]
        for layer in child.modules():
            if isinstance(layer, nn.Sequential):
                layer = layer[0]
            
            if isinstance(layer, nn.Linear):
                (W_1, in_size) = list(net.parameters())[0].shape
                dummy_input = torch.randn((batch_size, in_size))
                output = net(dummy_input)
            elif isinstance(layer, nn.Conv2d):
                (K, in_C, K_h, K_w) = list(net.parameters())[0].shape
                if input_dim == None:
                    dummy_input = torch.randn(batch_size, in_C, 1024, 1024)
                    print('Since the input dimension was not specified the default input size was set to be (1024, 1024)')
                else:
                    if input_dim[0] != in_C:
                        error_str = 'input channels {} doesnt match the input channel size of network {}'.format(input_dim[0], in_C)
                        raise Exception(error_str)
                    dummy_input = torch.randn(batch_size, in_C, input_dim[1], input_dim[2])
                
                output = net(dummy_input)
            break
        break
    
    return output.shape


class Vocab():
    """Vocabulary for text."""
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        # Flatten a 2D list if needed
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        counter = collections.Counter(tokens)

        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        # The list of unique tokens
        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq])))
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)
    
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    
    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]

    @property
    def unk(self):  # Index for the unknown token
        return self.token_to_idx['<unk>']
    
    