import time
from tqdm import tqdm
import argparse


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from configs import *
from datasets import get_dataloaders
from models import model_trainers
from utils import (criterions,
                   all_metrics,
                   trainLogging,
                   save_checkpoint)

from validation import Validator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices = ['train', 'validation', 'test'], help = 'choose between train, validation, test')
    parser.add_argument('--version', type = int, help = 'version of the config')
    parser.add_argument('--dataset', help = 'name of the dataset')

    parser.add_argument('--data-dir', help = 'path to the dataset dir')
    parser.add_argument('--json-path', help = 'path to the json data of dataset')
    parser.add_argument('--split', choices = ['training', 'validation', 'inference'], help = "choose between 'training', 'validation', 'inference'")
    parser.add_argument('--image-size', type = int, nargs = 2, help = 'dimensions of image') 
    parser.add_argument('--labels', nargs = '*', help = 'list of labels to be segmented')
    parser.add_argument('--window', type = int, nargs = 2, help = 'windowing of image')
    parser.add_argument('--window-preset', choices = ['ct_abdomen','ct_liver','ct_spleen','ct_pancreas'], help = 'choose between window presets')
    parser.add_argument('--transform', action = 'store_true', help = 'apply transformations to the image')

    parser.add_argument('--model', choices = ['unet', 'hc_unet'], help = 'choose between unet and hc_unet')
    parser.add_argument('--loss', help = 'loss function to be used',
                        choices=['dice', 'cross_entropy', 'jaccard', 'hyperul', 'hyperbolicdistance'])
    parser.add_argument('--loss-list', nargs = '*', help = 'list of loss functions to be used',
                        choices=['dice', 'cross_entropy', 'jaccard', 'hyperul', 'hyperbolicdistance'])
    parser.add_argument('--weights', nargs = '*', type = float, help = 'list of weights for loss functions')
    parser.add_argument('--lr', type = float, help = 'learning rate')

    parser.add_argument('--metric', choices = ['all', 'miou', 'precision', 'recall', 'dice'], help = 'choose between metrics')
    parser.add_argument('--batch-size', type = int, help = 'batch size')
    parser.add_argument('--epochs', type = int, help = 'number of epochs')
    parser.add_argument('--checkpoint-dir', help = 'path to the checkpoint dir')
    parser.add_argument('--log-dir', help = 'path to the log dir')


    return parser.parse_args()

class Trainer:
    def __init__(self, train_data : DataLoader,
                 trainer, epochs: int,
                 validator,
                 criterion, metrics,
                 gpu_id: int,
                 train_logger):
        
        self.gpu_id = gpu_id
        
        self.data = train_data
        self.model : nn.Module = trainer.model
        self.criterion = criterion
        self.metrics = metrics
        self.optimizers = trainer.optimizers
        self.epochs = epochs

        self.model = DDP(self.model, device_ids = [gpu_id])

        self.validator = validator
        self.best_val_dice = 0
        self.logger = train_logger
    
    def _run_epoch(self):

        # Initialize epoch logs to 0
        epoch_logs : dict = {
            'train_loss' : 0,
            **{metric.name: 0 for metric in self.metrics}
        }

        # Train on all Batches
        for inputs, masks in self.data:
            inputs = inputs.to(self.gpu_id)
            masks = masks.to(self.gpu_id)

            loss, metrics = self._run_batch(inputs, masks)

            # Accumulate Logs
            epoch_logs['train_loss'] += loss
            for metric in metrics:
                epoch_logs[metric.name] += metrics[metric]

        # Compute Average for all logs 
        for log in epoch_logs:
            epoch_logs[log] /= len(self.data)
        
        # self.logger.add_epoch_logs(
        #     epoch, 
        # )

    def _run_batch(self, inputs, masks):

        for optimizer in self.optimizers:
            optimizer.zero_grad()

        outputs = self.model(inputs)
        loss = self.criterion(outputs, masks)

        loss.backward()

        for optimizer in self.optimizers:
            optimizer.step()

        outputs = torch.argmax(outputs, dim = 1)

        metrics = {}
        for metric in self.metrics:
            metrics[metric.name] += metric.compute(outputs, masks,
                                                   self.data.dataset.label_to_pixel_value)[1]
            
        return loss.item(), metrics
            
    def train(self):
        start_time = time.time()
        self.model.train()

        for epoch in self.epochs:
            self._run_epoch(epoch)

            # Validation
            self.validator.validate()

        training_time = time.time() - start_time
            
        if self.gpu_id == 0:
            print(f"Training Complete in {training_time:.2f}s with {training_time/self.epochs:.2f} for epcoh")

        



    
def main():
    args = parse_args().__dict__

    all_config = allConfig(**args)

    config_filename = all_config.get_config_filename()
    all_config.save_config(config_filename)

    train_config = trainConfig(**args)

    trainDataloader, valDataloader = get_dataloaders(config = amosDatasetConfig(**args))

    # Model Initialization
    model_trainer = model_trainers[train_config.model]()

    # Loss Initialization
    if len(args.loss_list) is not None:
        criterion = criterions['combined'](
            loss_list = train_config.loss_list,
            weights = train_config.weights
        )
        train_config.loss = 'combined'
    else:
        criterion = criterions[train_config.loss]()
    
    # Metric Initialization
    if train_config.metric == 'all':
            metrics = [metric() for metric in all_metrics.values()]
    else :
        metrics = [all_metrics[train_config.metric]]

    trainer = Trainer(
        train_data = trainDataloader,
        trainer = model_trainer,
        epochs = train_config.epochs,
        criterion = criterion,
        metrics = metrics,
        train_logger = logger,
        # gpu_id = 0
    )
    # Logger Initialization
    logger = trainLogging(metrics = [metric.name for metric in metrics], config = train_config)

    
    # Save Logs
    logger.save_train_logs(filename = train_config.log_dir + config_filename + '.csv')

    # print(f"Training Completed in {total_training_time:.2f} seconds")

if __name__ == "__main__":
    main()