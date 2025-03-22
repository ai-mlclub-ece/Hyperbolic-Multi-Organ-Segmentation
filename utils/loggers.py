import pandas as pd
from configs import Config, trainConfig

class trainLogging:
    def __init__(self, metrics: list[str], config: Config = trainConfig()):

        self.config = config

        columns = ['epoch', 'epoch_time', 'train_loss', 'val_loss']

        train_metric_cols = ['train_' + metric for metric in metrics]
        val_metric_cols = ['val_' + metric for metric in metrics]

        columns = columns + train_metric_cols + val_metric_cols
        self.logs: pd.DataFrame = pd.DataFrame(columns = columns)

    def add_epoch_logs(self, epoch, epoch_time, train_loss, train_metrics, val_loss, val_metrics):
        
        epoch_row = {
            'epoch': epoch,
            'epoch_time': epoch_time,
            'train_loss': train_loss,
            'val_loss': val_loss
        }

        epoch_row.update({f'train_{metric}': value for metric, value in train_metrics.items()})
        epoch_row.update({f'val_{metric}' : value for metric, value in val_metrics.items()})

        self.logs = self.logs.append(epoch_row, ignore_index=True)

    def save_train_logs(self, filename: str = None):
        self.logs = self.logs.sort_values(by = 'epoch')
        self.logs.to_csv(filename, index=False)