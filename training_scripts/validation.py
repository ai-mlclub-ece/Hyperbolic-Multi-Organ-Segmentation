import torch
from torch.utils.data import DataLoader


class Validator:
    def __init__(self, val_data : DataLoader,
                 criterion, metrics, multi_gpu):
        self.gpu_id = int(os.environ["LOCAL_RANK"]) if multi_gpu else 0
        self.data = val_data
        self.criterion = criterion
        self.metrics = metrics
    
    def validate(self, model):

        # Initialize logs to 0
        logs : dict = {
            'loss' : 0,
            **{metric.name: 0 for metric in self.metrics}
        }

        model.eval()
        
        with torch.no_grad():
        # Validate on all Batches
            for inputs, masks in self.data:
                inputs = inputs.to(self.gpu_id)
                masks = masks.to(self.gpu_id)

                loss, metrics = self._run_batch(model, inputs, masks)

                # Accumulate Logs
                logs['loss'] += loss
                for metric in metrics:
                    logs[metric] += metrics[metric]

        # Compute Average for all logs 
        for log in logs:
            logs[log] /= len(self.data)

        return logs

    def _run_batch(self, model, inputs, masks):

        outputs = model(inputs)
        loss = self.criterion(outputs, masks)

        outputs = torch.argmax(outputs, dim = 1)

        metrics = {metric.name: 0 for metric in self.metrics}
        for metric in self.metrics:
            metrics[metric.name] += metric.compute(outputs, masks.squeeze(1))[1]
            
        return loss.item(), metrics