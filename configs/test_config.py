from config import Config

class testConfig(Config):
    def __init__(self, **args):
        super().__init__(**args)

        self.model: str = ''
        self.checkpoint: str = ''
        self.batch_size: int = 0
        self.save_results: bool = False
        self.r
        esult_dir: str = ''

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


