
import os
import sys
import time
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

    parser.add_argument('--data-dir', type = str, help = 'path to the dataset dir')
    parser.add_argument('--split', type = str, choices = ['training', 'validation', 'inference'], help = "choose between 'training', 'validation', 'inference'")
    parser.add_argument('--image-size', type = int, nargs = 2, help = 'dimensions of image') 
    parser.add_argument('--labels', type = str, nargs = '*', help = 'list of labels to be segmented')
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
    parser.add_argument('--checkpoint-dir', type = str, help = 'path to the checkpoint dir')
    parser.add_argument('--log-dir', type = str, help = 'path to the log dir')

    parser.add_argument('--single-gpu', action= 'store_true', help= 'use single gpu for training')


    return parser.parse_args()

class Trainer:
    def __init__(self, train_data : DataLoader,
                 trainer, epochs: int,
                 validator, val_data : DataLoader,
                 criterion, metrics,
                 multi_gpu: bool,
                 train_logger,
                 config_filename,
                 train_config):
        
        self.config = train_config
        self.gpu_id = int(os.environ["LOCAL_RANK"]) if multi_gpu else 0
        
        self.data = train_data
        self.model : nn.Module = trainer.model.to(self.gpu_id)
        self.criterion = criterion
        self.metrics = metrics
        self.optimizers = trainer.optimizers
        self.epochs = epochs

        self.multi_gpu = multi_gpu
        if multi_gpu:
            self.model = DDP(self.model, device_ids = [self.gpu_id])

        self.validator = validator(
            val_data = val_data,
            criterion = criterion,
            metrics = metrics
        )
        self.best_val_dice = 0
        self.logger = train_logger
        self.filename = config_filename
    
    def _run_epoch(self, epoch : int) -> dict:

        # Initialize epoch logs to 0
        epoch_logs : dict = {
            'loss' : 0,
            **{metric.name: 0 for metric in self.metrics}
        }

        train_iterator = tqdm(self.data, total = len(self.data), desc = f"Epoch {epoch + 1}")
        # Train on all Batches
        for inputs, masks in train_iterator:
            inputs = inputs.to(self.gpu_id)
            masks = masks.to(self.gpu_id)

            loss, metrics = self._run_batch(inputs, masks)

            # Accumulate Logs
            epoch_logs['loss'] += loss
            for metric in metrics:
                epoch_logs[metric] += metrics[metric]

        # Compute Average for all logs 
        for log in epoch_logs:
            epoch_logs[log] /= len(self.data)
        
        return epoch_logs

    def _run_batch(self, inputs, masks):

        for optimizer in self.optimizers:
            optimizer.zero_grad()

        outputs = self.model(inputs)
        loss = self.criterion(outputs, masks)

        loss.backward()

        for optimizer in self.optimizers:
            optimizer.step()

        outputs = torch.argmax(outputs, dim = 1)

        metrics = {metric.name: 0 for metric in self.metrics}
        for metric in self.metrics:
            metrics[metric.name] += metric.compute(outputs, masks.squeeze(1),
                                                   self.data.dataset.label_to_pixel_value)[1]
            
        return loss.item(), metrics
            
    def train(self):
        start_time = time.time()
        self.model.train()

        if self.gpu_id == 0:
            print(f"Training {self.config.model} for {self.epochs} epochs on {torch.cuda.device_count()} GPUs")
        for epoch in range(self.epochs):
            epoch_logs = self._run_epoch(epoch)

            # Validation
            val_logs = self.validator.validate(model = self.model)

            # Update Logs
            self.logger.add_epoch_logs(epoch, epoch_logs, val_logs)

            # Save if best model checkpoint
            if val_logs['dice_score'] > self.best_val_dice:
                save_checkpoint(self.model, self.optimizers, epoch,
                                self.checkpoint_dir + self.filename + '.pth',
                                self.multi_gpu)
        # Save Logs
        self.logger.save_train_logs(filename = self.train_config.log_dir + self.filename + '.csv')

        training_time = time.time() - start_time
            
        if self.gpu_id == 0:
            print(f"Training Complete in {training_time:.2f}s with {training_time/self.epochs:.2f} for epcoh")
            print(f"Best Validation Dice Score: {self.best_val_dice:.4f}")
        



    
def main():
    args = parse_args().__dict__

    if args['data_dir'] is None or '':
        raise Exception("Data Directory is not provided")

    if not torch.cuda.is_available():
        raise Exception("Cuda is not available, training on CPU is not Ideal")
        

    all_config = allConfig(**args)

    config_filename = all_config.get_config_filename()
    all_config.save_config(args['data_dir'] + config_filename)

    train_config = trainConfig(**args)

    multi_gpu = not args['single_gpu']

    if torch.cuda.device_count() == 1:
        multi_gpu = False


    trainDataloader, valDataloader = get_dataloaders(multi_gpu = multi_gpu,
                                                     config = amosDatasetConfig(**args))

    # Model Initialization
    model_trainer = model_trainers[train_config.model]()

    # Loss Initialization
    if args['loss_list'] is not None:
        criterion = criterions['combined'](
            loss_list = train_config.loss_list,
            weights = train_config.weights,
            labels_to_pixels = trainDataloader.dataset.label_to_pixel_value
        )
        train_config.loss = 'combined'
    else:
        criterion = criterions[train_config.loss](labels_to_pixels = trainDataloader.dataset.label_to_pixel_value)
    
    # Metric Initialization
    if train_config.metric == 'all':
            metrics = [metric() for metric in all_metrics.values()]
    else :
        metrics = [all_metrics[train_config.metric]]

    # Logger Initialization
    logger = trainLogging(metrics = [metric.name for metric in metrics], config = train_config)

    trainer = Trainer(
        train_data = trainDataloader,
        validator = Validator,
        val_data = valDataloader,
        trainer = model_trainer,
        epochs = train_config.epochs,
        criterion = criterion,
        metrics = metrics,
        train_logger = logger,
        config_filename = config_filename,
        multi_gpu = multi_gpu,
        train_config = train_config
    )

    trainer.train()
    
if __name__ == "__main__":
    main()