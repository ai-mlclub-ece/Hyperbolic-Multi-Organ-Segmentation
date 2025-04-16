import torch
from torch import nn

import pandas as pd

def save_checkpoint(model: nn.Module, optimizers, schedulers, epoch: int, filename: str, best_val_dice: float, multi_gpu: bool):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if multi_gpu else model.state_dict(),
        'optimizers_state_dict': [optimizer.state_dict() for optimizer in optimizers],
        'schedulers_state_dict': [scheduler.state_dict() for scheduler in schedulers],
        'best_val_dice': best_val_dice
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model: nn.Module, optimizers, schedulers, filename: str):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    best_val_dice = checkpoint['best_val_dice']
    
    for optimizer, state_dict in zip(optimizers, checkpoint['optimizers_state_dict']):
        optimizer.load_state_dict(state_dict)
    
    for scheduler, state_dict in zip(schedulers, checkpoint['schedulers_state_dict']):
        scheduler.load_state_dict(state_dict)
    
    return model, optimizers, schedulers, epoch, best_val_dice