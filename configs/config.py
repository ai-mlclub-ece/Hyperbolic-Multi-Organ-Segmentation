import json

class Config:
    def __init__(self, **args):

        self.set_default()
        self.set_args(**args)
    
    def set_default(self):
        pass

    def set_args(self, **args):
        for key, value in args.items():
            if hasattr(self, key) & value is not None:
                setattr(self, key, value)

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
        self.loss = 'dice'
        self.weights = [1.0]
        self.loss_list = [self.loss]


    
if __name__ == '__main__':

    config = Config()
    print(config.__dict__)