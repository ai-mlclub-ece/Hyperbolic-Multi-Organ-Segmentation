import os
import json

class Config:
    def __init__(self, **args):
        self.version: int = 0
        self.dataset: str = ''

        self.set_default()
        self.set_args(**args)
    
    def set_default(self):
        self.version = 1
        self.dataset = 'AMOS Dataset'

    def set_args(self, **args):
        for key, value in args.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid argument: {key}")

    def save_config(self, filename: str):
        """
        Save the configuration to a Json file
        """
        if filename is None:
            filename = self.get_config_filename()
        with open(f"{filename}.json", 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def get_config_filename(self):
        return f"config_{self.version}"
    
class trainConfig(Config):
    def __init__(self, **args):
        super().__init__(**args)

        self.model: str = ''
        self.loss: str = ''
        self.loss_config: dict = lossConfig(**args).__dict__
        self.metric: str = ''
        self.optimizer: str = ''
        self.learning_rate: float = 0.0
        self.batch_size: int = 0
        self.epochs: int = 0
        self.image_size: tuple[int, int] = (0, 0)
        self.checkpoint_dir: str = ''
        self.log_dir: str = ''

        self.set_default()
        self.set_args(**args)

    def set_default(self):
        super().set_default()
        self.model = 'UNet'
        self.loss = 'bce'
        self.loss_config = lossConfig().set_default().__dict__
        self.metric = 'all'
        self.optimizer = 'adam'
        self.learning_rate = 10e-4
        self.batch_size = 16
        self.epochs = 10
        self.image_size: tuple[int, int] = (512, 512)
        self.checkpoint_dir = 'checkpoints'
        self.log_dir = 'logs'


    def get_config_filename(self):
        filename = f"{self.model}_{self.loss}-v{self.version}"
        if os.path.exists(f"{filename}.json"):
            filename = f"{self.model}_{self.loss}-v{self.version + 1}"
        return filename

    
class lossConfig(Config):
    def __init__(self, **args):
        super().__init__(**args)

        self.loss: str = ''
        self.weights: list[float] = []
        self.loss_list: list[str] = [self.loss]

        self.set_default()
        self.set_args(**args)

    def set_default(self):
        super().set_default()
        self.loss = 'bce'
        self.weights = [1.0]
        self.loss_list = [self.loss]

class testConfig(Config):
    def __init__(self, **args):
        super().__init__(**args)

        self.model: str = ''
        self.checkpoint: str = ''
        self.batch_size: int = 0
        self.save_results: bool = False
        self.result_dir: str = ''

        self.set_default()
        self.set_args(**args)

    def set_default(self):
        super().set_default()
        self.model = 'UNet'
        self.checkpoint = 'UNet-bce-v1.pth'
        self.batch_size = 16
        self.save_results = False
        self.result_dir = self.get_config_filename()
    
    def get_config_filename(self):
        filename = f"{self.checkpoint.split('.')[0]}"
        return filename

    
if __name__ == '__main__':

    config = trainConfig(model = 'hc_unet')
    print(config.__dict__)