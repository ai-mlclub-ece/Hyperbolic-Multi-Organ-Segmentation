import pandas as pd
from configs import Config, trainConfig

class trainLogging:
    def __init__(self, metrics: list[str]):

        columns = ['epoch', 'train_loss', 'val_loss']

        train_metric_cols = ['train_' + metric for metric in metrics]
        val_metric_cols = ['val_' + metric for metric in metrics]

        columns = columns + train_metric_cols + val_metric_cols
        self.logs: pd.DataFrame = pd.DataFrame(columns = columns)

    def add_epoch_logs(self, epoch, train_logs, val_logs):
        
        epoch_row = {'epoch': epoch}

        epoch_row.update({f'train_{metric}': value for metric, value in train_logs.items()})
        epoch_row.update({f'val_{metric}' : value for metric, value in val_logs.items()})

        if self.logs.empty:
            self.logs = pd.DataFrame([epoch_row])
        else:
            self.logs = pd.concat([self.logs, pd.DataFrame([epoch_row])], ignore_index=True)

    def save_train_logs(self, filename: str = None):
        self.logs = self.logs.sort_values(by = 'epoch')
        self.logs.to_csv(filename, index=False)