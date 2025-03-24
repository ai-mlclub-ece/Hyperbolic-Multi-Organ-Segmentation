import os
import json

from .config import Config, lossConfig
from .dataset_config import amosDatasetConfig
from .train_config import trainConfig
from .test_config import testConfig
from .models_config import unetConfig, hc_unetConfig

class allConfig(Config):
    def __init__(self, **args):
        super().__init__(**args)

        self.version: int = 0
        self.all_configs_dir: str = ''
        self.dataset_config = amosDatasetConfig(**args).__dict__
        self.train_config = trainConfig(**args).__dict__
        self.test_config = testConfig(**args).__dict__
        self.unet_config = unetConfig(**args).__dict__
        self.hc_unet_config = hc_unetConfig(**args).__dict__
        self.loss_config = lossConfig(**args).__dict__

        self.set_default()
        self.set_args(**args)

        if not os.path.exists(self.all_configs_dir):
            os.makedirs(self.all_configs_dir)

    def set_default(self):
        super().set_default()
        self.version = 1
        self.all_configs_dir: str = 'all_configs/' 

    def get_config_filename(self):
        filename = f"{self.train_config['model']}_{self.train_config['loss']}-v{self.version}"
        if os.path.exists(self.all_configs_dir + f"{filename}.json"):
            filename = f"{self.train_config["model"]}_{self.train_config["loss"]}-v{self.version + 1}"
        return filename
