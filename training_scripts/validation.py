from configs import *

import torch
from torch.utils.data import DataLoader
from datasets import get_dataloaders
import tqdm

from models import trainers
from utils import criterions, all_metrics



def validate(valDataloader: DataLoader, trainer, config):


    if len(config.loss_list) is not None:
        criterion = criterions['combined'](
            loss_list = config.loss_list,
            weights = config.weights
        )
    else:
        criterion = criterions[config.loss]()

    with torch.no_grad():
        trainer.model.eval()


        if config.metric == 'all':
            metrics = [metric() for metric in all_metrics.values()]
        else :
            metrics = [all_metrics[config.metric]]

        val_metrics = {metric.name: 0 for metric in metrics}
        val_loss    = 0

        val_iterator = tqdm(enumerate(valDataloader), total = len(valDataloader), desc = f'Epoch-{epoch+1}:')

        for i, (inputs, masks) in val_iterator:
            inputs = inputs
            masks = masks

            outputs = trainer.model(inputs)
            loss = criterion(outputs, masks)


            val_loss += loss.item()
            for metric in metrics:
                val_metrics[metric.name] += metric.compute(outputs, masks,
                                                             valDataloader.dataset.label_to_pixel_value)

            val_iterator.set_postfix({
                f'{loss.name}_loss' : f'{val_loss/(i+1):.4f}',
                **{
                    metric.name : val_metrics[metric.name] for metric in metrics
                }
            })
        
        val_loss /= len(valDataloader)
        val_metrics = {
            metric.name : value/len(valDataloader) for metric, value in val_metrics.items()
        }

    return val_loss, val_metrics
