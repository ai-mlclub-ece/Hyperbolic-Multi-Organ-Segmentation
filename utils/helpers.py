import torch
from torch import nn

import pandas as pd

def save_checkpoint(model: nn.Module, optimizers, epoch: int, filename: str, multi_gpu):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if multi_gpu else model.state_dict(),
        'optimizers_state_dict': [optimizer.state_dict() for optimizer in optimizers]
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model: nn.Module, optimizers, filename: str):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    for optimizer, state_dict in zip(optimizers, checkpoint['optimizers_state_dict']):
        optimizer.load_state_dict(state_dict)
    return model, optimizers, epoch