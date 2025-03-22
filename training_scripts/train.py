import argparse
from configs import *

import torch
from torch.utils.data import DataLoader
from datasets import get_dataloaders
import tqdm

from models import trainers
from utils import criterions, all_metrics

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

def main():
    args = parse_args().__dict__
    train_config = trainConfig(**args)

    trainDataloader, valDataloader = get_dataloaders(config = amosDatasetConfig(**args))

    trainer = trainers[train_config.model]()
    if len(args.loss_list) is not None:
        criterion = criterions['combined'](
            loss_list = train_config.loss_list,
            weights = train_config.weights
        )
    else:
        criterion = criterions[train_config.loss]()


    for epoch in train_config.epochs:
        trainer.model.train()


        if train_config.metric == 'all':
            metrics = [metric() for metric in all_metrics.values()]
        else :
            metrics = [all_metrics[train_config.metric]]

        train_metrics = {metric.name: 0 for metric in metrics}
        train_loss    = 0

        train_iterator = tqdm(enumerate(trainDataloader), total = len(trainDataloader), desc = f'Epoch-{epoch+1}:')

        for i, (inputs, masks) in train_iterator:
            inputs = inputs
            masks = masks

            for optimizer in trainer.opttimizers:
                optimizer.zero_grad()

            outputs = trainer.model(inputs)
            loss = criterion(outputs, masks)

            loss.backward()

            for optimizer in trainer.opttimizers:
                optimizer.step()

            train_loss += loss.item()
            for metric in metrics:
                train_metrics[metric.name] += metric.compute(outputs, masks,
                                                             trainDataloader.dataset.label_to_pixel_value)

            train_iterator.set_postfix({
                f'{loss.name}_loss' : f'{train_loss/(i+1):.4f}',
                **{
                    metric.name : train_metrics[metric.name] for metric in metrics
                }
            })
        
        train_metrics = {
            metric.name : value/len(trainDataloader) for metric, value in train_metrics.items()
        }

    
