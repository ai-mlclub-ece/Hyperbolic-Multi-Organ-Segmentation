from .amos_dataset import AmosDataset
from torch.utils.data import DataLoader
from configs import Config, amosDatasetConfig

def get_dataloaders(config: Config = amosDatasetConfig()) -> list[DataLoader]:
    
    if config.mode == 'train':
        trainDataloader = DataLoader(
            AmosDataset(**config.__dict__),
            batch_size= config.batch_size
        )
        val_dict = config.__dict__['split'] = 'validation'

        valDataloader = DataLoader(
            AmosDataset(**val_dict),
            batch_size= config.batch_size
        )
        return trainDataloader, valDataloader
    elif config.mode == 'validation' :
        val_dict = config.__dict__['split'] = 'validation'

        valDataloader = DataLoader(
            AmosDataset(**val_dict),
            batch_size= config.batch_size
        )
        return valDataloader
    
    else :
        test_dict = config.__dict__['split'] = 'test'

        testDataloader = DataLoader(
            AmosDataset(**test_dict),
            batch_size= config.batch_size
        )
        return testDataloader
    
if __name__ == '__main__':
    dataset = AmosDataset