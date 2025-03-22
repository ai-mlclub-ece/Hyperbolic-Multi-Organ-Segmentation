from config import Config
from config import lossConfig
from dataset_config import amosDatasetConfig
from models_config import unetConfig, hc_unetConfig

import os


class trainConfig(Config,
                  lossConfig,
                  amosDatasetConfig, 
                  unetConfig,
                  hc_unetConfig):
    def __init__(self, **args):
        super().__init__(**args)

        self.model: str = ''
        self.metric: str = ''
        self.epochs: int = 0
        self.checkpoint_dir: str = ''
        self.log_dir: str = ''

        self.set_default()
        self.set_args(**args)

    def set_default(self):
        super().set_default()
        self.model = 'UNet'
        self.metric = 'all'
        self.epochs = 10
        self.checkpoint_dir = 'checkpoints'
        self.log_dir = 'logs'


    def get_config_filename(self):
        filename = f"{self.model}_{self.loss}-v{self.version}"
        if os.path.exists(f"{filename}.json"):
            filename = f"{self.model}_{self.loss}-v{self.version + 1}"
        return filename

