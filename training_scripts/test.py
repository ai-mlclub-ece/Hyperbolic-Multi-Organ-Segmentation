import os
from utils import load_checkpoint
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset



class Tester:
    def __init__(self, test_data : Dataset, trainer, n_samples, batch_size,
                 criterion, metrics, checkpoint_path : str, random_seed = 42):
        
        torch.manual_seed(random_seed)

        self.gpu_id =  0
        self.data = self.get_subset(test_data, n_samples,batch_size)
        
        self.model = trainer.model.to(self.gpu_id)
        self.optimizers = trainer.optimizers
        self.criterion = criterion
        self.metrics = metrics

        self.checkpoint_path = checkpoint_path

        self.model, _, _ = load_checkpoint(
            self.model, self.optimizers, self.checkpoint_path
        )
    def get_subset(self, test_data:Dataset, n_samples:int, batch_size:int):
        samples = torch.randint(0,len(test_data), (n_samples,)).tolist()
        data = Subset(test_data,samples)
        return DataLoader(data,batch_size,shuffle = False)
        
    def infer(self) -> tuple[dict, np.ndarray]:

        # Initialize logs to 0
        logs : dict = {
            'loss' : 0,
            **{metric.name: 0 for metric in self.metrics}
        }

        self.model.eval()

        all_images = []
        all_masks = []
        all_preds = []

        with torch.no_grad():
        # Validate on all Batches
            for inputs, masks in self.data:
                inputs = inputs.to(self.gpu_id)
                masks = masks.to(self.gpu_id)

                preds, loss, metrics = self._run_batch(self.model, inputs, masks)
                all_preds.append(preds.cpu())
                all_images.append(inputs.cpu())
                all_masks.append(masks.cpu())

                # Accumulate Logs
                logs['loss'] += loss
                for metric in metrics:
                    logs[metric] += metrics[metric]

        # Compute Average for all logs 
        for log in logs:
            logs[log] /= len(self.data)

        return logs, torch.cat(all_images).numpy(), torch.cat(all_masks).numpy(), torch.cat(all_preds).numpy()

    def _run_batch(self, model, inputs, masks):

        outputs = model(inputs)
        loss = self.criterion(outputs, masks)

        outputs = torch.argmax(outputs, dim = 1)

        metrics = {metric.name: 0 for metric in self.metrics}
        for metric in self.metrics:
            metrics[metric.name] += metric.compute(outputs, masks.squeeze(1))[1]
            
        return outputs, loss.item(), metrics