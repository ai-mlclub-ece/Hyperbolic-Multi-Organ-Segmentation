from config import Config

class amosDatasetConfig(Config):
    def __init__(self, **args):
        super().__init__(**args)

        self.data_dir: str = ''
        self.jsonPath: str = ''
        self.split   : str = ''
        self.img_size: tuple[int,int] = ()
        self.labels  : list[str] = []
        self.window  : tuple[int,int] = ()
        self.window_preset : str = ''
        self.transform : bool = False

        self.set_default()
        self.set_args(**args)
        
    def set_default(self):
        super().set_default()

        self.data_dir: str = ''
        self.jsonPath: str = ''
        self.split   : str = 'training'
        self.img_size: tuple[int,int] = (512, 512)
        self.labels  : list[str] = ['liver', 'pancreas', 'spleen']
        self.window  : tuple[int,int] = None
        self.window_preset : str = 'ct_abdomen'
        self.transform : bool = False


        
