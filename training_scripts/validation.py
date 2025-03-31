import os
import torch
from torch.utils.data import DataLoader

class Validator:
    def __init__(self, val_data: DataLoader, criterion, metrics, multi_gpu: bool = False):
        
        self.gpu_id = int(os.environ["LOCAL_RANK"]) if multi_gpu else 0
        self.data = val_data
        self.criterion = criterion
        self.metrics = metrics

    def validate(self, model):
        
        logs: dict = {
            'loss': 0.0,
            **{metric.name: 0.0 for metric in self.metrics}
        }

        model.eval()

        with torch.no_grad():
            for inputs, masks in self.data:
                inputs = inputs.to(self.gpu_id)
                masks = masks.to(self.gpu_id)

                loss, batch_metrics = self._run_batch(model, inputs, masks)

                # Accumulate loss and metrics over batches
                logs['loss'] += loss
                for metric_name, metric_value in batch_metrics.items():
                    logs[metric_name] += metric_value

        # Average logs across all batches
        num_batches = len(self.data)
        for key in logs:
            logs[key] /= num_batches

        return logs

    def _run_batch(self, model, inputs, masks):
        outputs = model(inputs)
        loss = self.criterion(outputs, masks)

        predictions = torch.argmax(outputs, dim=1)

        metrics_result = {}
        for metric in self.metrics:
            _, value = metric.compute(predictions, masks.squeeze(1))
            metrics_result[metric.name] = value

        return loss.item(), metrics_result