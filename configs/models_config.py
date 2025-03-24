from .config import Config
from .dataset_config import amosDatasetConfig
from torch.optim import Adam
import geoopt


class unetConfig(amosDatasetConfig):
    def __init__(self, **args):
        super().__init__(**args)

        self.optimizer = None
        self.learning_rate: float = 0.0

        self.set_default()
        self.set_args(**args)

    def set_default(self):
        super().set_default()

        self.optimizer = 'Adam'
        self.learning_rate: float = 10e-4


class hc_unetConfig(amosDatasetConfig):
    def __init__(self, **args):
        super().__init__(**args)

        self.optimizers: list = None
        self.learning_rate: float = 0.0
        self.embedding_dim: int = None
        self.curvature: float = 0.1
        self.lambda_cp: float = 1
         
        self.set_default()
        self.set_args(**args)

    def set_default(self):
        super().set_default()

        self.optimizers: list = ['Adam', 'RiemannianAdam']
        self.learning_rate: float = 10e-4
        self.embedding_dim: int = 256
        self.curvature: float = 0.1
        self.lambda_cp: float = 1
