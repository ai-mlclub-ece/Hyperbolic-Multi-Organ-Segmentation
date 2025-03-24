from .amos_dataset import AmosDataset
from torch.utils.data import DataLoader
from configs import Config, amosDatasetConfig

from torch.utils.data.distributed import DistributedSampler

def get_dataloaders(gpus : int, config: Config = amosDatasetConfig()) -> list[DataLoader]:
    
    if config.mode == 'train':
        
        train_dataset = AmosDataset(**config.__dict__)

        val_dict = config.__dict__['split'] = 'validation'
        val_dataset = AmosDataset(**val_dict)

        trainDataloader = DataLoader(
            train_dataset,
            batch_size= config.batch_size,
            shuffle = False if gpus > 1 else True,
            sampler = DistributedSampler(train_dataset) if gpus > 1 else None
        )
        

        valDataloader = DataLoader(
            val_dataset,
            batch_size= config.batch_size,
            shuffle = False,
            sampler = DistributedSampler(train_dataset) if gpus > 1 else None
        )

        return trainDataloader, valDataloader
    
    elif config.mode == 'validation' :

        val_dict = config.__dict__['split'] = 'validation'
        val_dataset = AmosDataset(**val_dict)

        valDataloader = DataLoader(
            val_dataset,
            batch_size= config.batch_size,
            shuffle = False,
            sampler = DistributedSampler(train_dataset) if gpus > 1 else None
        )
        return valDataloader
    
    else :
        test_dict = config.__dict__['split'] = 'test'
        test_dataset = AmosDataset(**test_dict)

        testDataloader = DataLoader(
            test_dataset,
            batch_size= config.batch_size,
            sampler = DistributedSampler(train_dataset) if gpus > 1 else None
        )
        return testDataloader
    
if __name__ == '__main__':
    pass