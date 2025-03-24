import os

from .amos_dataset import AMOS_Dataset
from torch.utils.data import DataLoader
from configs import Config, amosDatasetConfig

from torch.utils.data.distributed import DistributedSampler

def get_dataloaders(multi_gpu : int, config: Config = amosDatasetConfig()) -> list[DataLoader]:
    
    if config.mode == 'train':
        train_dict = config.__dict__
        train_dict['json_path'] = os.path.join(config.data_dir, 'dataset.json')
        
        train_dataset = AMOS_Dataset(**train_dict)

        val_dict = config.__dict__
        val_dict['split'] = 'validation'

        val_dataset = AMOS_Dataset(**val_dict)

        trainDataloader = DataLoader(
            train_dataset,
            batch_size= config.batch_size,
            shuffle = False if multi_gpu else True,
            sampler = DistributedSampler(train_dataset) if multi_gpu else None
        )
        

        valDataloader = DataLoader(
            val_dataset,
            batch_size= config.batch_size,
            shuffle = False,
            sampler = DistributedSampler(train_dataset) if multi_gpu else None
        )

        return trainDataloader, valDataloader
    
    elif config.mode == 'validation' :

        val_dict = config.__dict__
        val_dict['split'] = 'validation'

        val_dataset = AMOS_Dataset(**val_dict)

        valDataloader = DataLoader(
            val_dataset,
            batch_size= config.batch_size,
            shuffle = False,
            sampler = DistributedSampler(train_dataset) if multi_gpu else None
        )
        return valDataloader
    
    else :
        test_dict = config.__dict__
        test_dict['split'] = 'test'

        test_dataset = AMOS_Dataset(**test_dict)

        testDataloader = DataLoader(
            test_dataset,
            batch_size= config.batch_size,
            sampler = DistributedSampler(train_dataset) if multi_gpu else None
        )
        return testDataloader
    
if __name__ == '__main__':
    pass