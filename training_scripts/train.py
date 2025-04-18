import os
import sys
import time
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import *
from datasets import get_dataloaders
from models import model_trainers
from utils import (criterions,
                   all_metrics,
                   trainLogging,
                   save_checkpoint, load_checkpoint,
                   trainLogVisualizer,
                   inferVisualizer,
                   schedulers)

from validation import Validator
from test import Tester

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices = ['train', 'validation', 'test'], help = 'choose between train, validation, test')
    parser.add_argument('--version', type = int, help = 'version of the config')
    parser.add_argument('--dataset', help = 'name of the dataset')
    parser.add_argument('--config-filename', type=str, help='Custom config filename (without extension). If not provided, will use model-loss-v1 format')

    parser.add_argument('--data-dir', type = str, help = 'path to the dataset dir')
    parser.add_argument('--all-configs-dir', type = str, help = 'path to the all configs dir')
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
        self.optimizers = trainer.optimizers
        
        # Add learning rate scheduler
        self.schedulers = []
        for optimizer in self.optimizers:
            scheduler = schedulers.get_scheduler(
                optimizer,
                mode='min',
                factor=0.1,
                patience=5
            )
            self.schedulers.append(scheduler)
            
        self.best_val_dice = 0
        epoch = 0
        # Load Checkpoint if exists
        if os.path.exists(train_config.checkpoint_dir + '/best_model.pth'):
            if self.gpu_id == 0:
                print(f"Loading model from {train_config.checkpoint_dir + '/best_model.pth'}")

            self.model, self.optimizers, self.schedulers, epoch, self.best_val_dice = load_checkpoint(
                model = self.model,
                optimizers = self.optimizers,
                schedulers = self.schedulers,
                filename = train_config.checkpoint_dir + '/best_model.pth'
            )
        
        self.criterion = criterion
        self.metrics = metrics
        self.epochs = epochs - epoch

        self.multi_gpu = multi_gpu
        if multi_gpu:
            self.model = DDP(self.model, device_ids = [self.gpu_id])

        self.validator = validator(
            val_data = val_data,
            criterion = criterion,
            metrics = metrics,
            multi_gpu = multi_gpu
        )
        self.logger = train_logger
        self.filename = config_filename
        
        # Gradient clipping threshold
        self.max_grad_norm = 1.0
    
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

        outputs, logits = self.model(inputs)
        loss = self.criterion(logits, masks)

        loss.backward()

        # Apply gradient clipping
        if self.multi_gpu:
            # For DDP, we need to access the module
            clip_grad_norm_(self.model.module.parameters(), self.max_grad_norm)
        else:
            clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        for optimizer in self.optimizers:
            optimizer.step()

        outputs = torch.argmax(outputs, dim = 1)

        metrics = {metric.name: 0 for metric in self.metrics}
        for metric in self.metrics:
            metrics[metric.name] += metric.compute(outputs, masks.squeeze(1))[1]
            
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

            # Update learning rate based on validation loss
            for scheduler in self.schedulers:
                scheduler.step(val_logs['loss'])

            # Update Logs
            self.logger.add_epoch_logs(epoch, epoch_logs, val_logs)

            # Save if best model checkpoint
            if (val_logs['dice_score'] > self.best_val_dice) & (self.gpu_id == 0):
                save_checkpoint(
                    model = self.model,
                    optimizers = self.optimizers,
                    schedulers = self.schedulers,
                    epoch = epoch,
                    filename = self.config.checkpoint_dir + '/best_model.pth',
                    best_val_dice = self.best_val_dice,
                    multi_gpu = self.multi_gpu
                )
                self.best_val_dice = val_logs['dice_score']
                print(f"Best Model Saved with Validation Dice Score: {self.best_val_dice:.4f}")
            
        if self.gpu_id == 0:
            # Save Logs
            self.logger.save_train_logs(filename = self.config.checkpoint_dir + '/train_logs.csv')

            training_time = time.time() - start_time
            print(f"Training Complete in {training_time:.2f}s with {training_time/self.epochs:.2f} for epcoh")
            print(f"Best Validation Dice Score: {self.best_val_dice:.4f}")
        



    
def main():
    init_process_group(backend='nccl')
    gpu_id = int(os.environ['LOCAL_RANK'])

    args = parse_args().__dict__

    if args['data_dir'] is None or '':
        raise Exception("Data Directory is not provided")

    if not torch.cuda.is_available():
        raise Exception("Cuda is not available, training on CPU is not Ideal")
    
    if args['loss_list'] is not None:
        if len(args['loss_list']) != len(args['weights']):
            raise Exception("Length of loss list and weights should be same")
        
        args['loss'] = 'combined'
        
    all_config = allConfig(**args)

    # Use provided config filename or generate default
    if args.get('config_filename'):
        config_filename = args['config_filename']
    else:
        config_filename = all_config.get_config_filename()

    if gpu_id == 0:
        all_config.save_config(all_config.all_configs_dir + config_filename)

    checkpoint_dir = all_config.train_config['checkpoint_dir'] + config_filename
    os.makedirs(checkpoint_dir, exist_ok = True)

    if gpu_id == 0:
        all_config.save_config(checkpoint_dir + '/config')

    train_config = trainConfig(**args)
    train_config.checkpoint_dir = checkpoint_dir

    multi_gpu = not args['single_gpu']

    if torch.cuda.device_count() == 1:
        multi_gpu = False


    trainDataloader, valDataloader = get_dataloaders(multi_gpu = multi_gpu,
                                                     config = amosDatasetConfig(**args))

    labels = trainDataloader.dataset.labels
    labels_to_pixels = trainDataloader.dataset.label_to_pixel_value

    # Model Initialization
    model_trainer = model_trainers[train_config.model]()
    
    # Loss Initialization
    if args['loss_list'] is not None:
        criterion = criterions['combined'](
            loss_list = train_config.loss_list,
            weights = train_config.weights,
            labels = labels,
            labels_to_pixels = labels_to_pixels
        )
        train_config.loss = 'combined'
    else:
        criterion = criterions[train_config.loss](labels = labels, labels_to_pixels = labels_to_pixels)
    
    # Metric Initialization
    if train_config.metric == 'all':
            metrics = [metric(labels = labels, labels_to_pixels = labels_to_pixels) for metric in all_metrics.values()]
    else :
        metrics = [all_metrics[train_config.metric](labels = labels, labels_to_pixels = labels_to_pixels)]

    # Logger Initialization
    logger = trainLogging(metrics = [metric.name for metric in metrics])

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

    destroy_process_group()
    
if __name__ == "__main__":
    main()