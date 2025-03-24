from .config import lossConfig

import os


class trainConfig(lossConfig):
    def __init__(self, **args):
        super().__init__(**args)

        self.model: str = ''
        self.metric: str = ''
        self.epochs: int = 0
        self.multi_gpu : bool = None
        self.checkpoint_dir: str = ''
        self.log_dir: str = ''

        self.set_default()
        self.set_args(**args)
        
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def set_default(self):
        super().set_default()
        self.model = 'unet'
        self.metric = 'all'
        self.epochs = 10
        self.multi_gpu = False
        self.checkpoint_dir = 'checkpoints/'
        self.log_dir = 'logs/'


