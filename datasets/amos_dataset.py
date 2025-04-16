import os
import sys
import cv2
import json
import random
from tqdm import tqdm

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import nibabel as nib

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class dataIngestion:
    def __init__(self, jsonPath):
        self.jsonPath = jsonPath
        self.jsonData = None
        self.loadJsonData()

    def loadJsonData(self):
        """
        Load the json file containing the Dataset image and label paths
        """
        with open(self.jsonPath) as f:
            self.jsonData = json.load(f)

    def getLabels(self) -> dict:
        """
        Get the label to pixel value mapping of the dataset

        Returns:

            labels (dict): The label to pixel value mapping of the dataset
        """
        pixel_to_label = self.jsonData["labels"]
        label_to_pixel= {label: int(pixel_value) for pixel_value, label in pixel_to_label.items()}
        return label_to_pixel
    
    def loadVolume(self, path:str, transpose: bool = False) -> np.ndarray:
        """
        Load the volume file (.nii.gz) from the given path

        Args:

            path (str): The path to the volume file
            transpose (bool): Whether to transpose the volume or not

        Returns:

            volume (np.ndarray): The volume data as a numpy array
        """

        if not transpose:
            return nib.load(path).get_fdata()
        else:
            return nib.load(path).get_fdata().transpose(2, 1, 0)

    
    def getSliceinfo(self, data_dir: str, split: str = "training") -> pd.DataFrame:
        """
        Get the slice information of the given split of the dataset

        Args:

            data_dir (str): The path to the dataset directory
            split (str): The split of the dataset to load. Can be "training", "validation" or "test"
            dataframe_dir (str): The path to the directory where the dataframes are stored or to be stored

        Returns:

            sliceinfo (pd.DataFrame): A dataframe containing the slice information of the dataset
        """
        if os.path.exists(data_dir + f"{split}_sliceinfo.csv"):
            sliceinfo = pd.read_csv(data_dir + f"{split}_sliceinfo.csv")
            print(f"Loaded {split} set DataFrame...")
            return sliceinfo
        else:
            print(f"Loading {split} set...")

            sliceinfo = pd.DataFrame(columns=["imgPath", "labelPath", "slice_idx"])

            for i in tqdm(range(len(self.jsonData[split]))):

                imgPath = self.jsonData[split][i]["image"]
                imgs = self.loadVolume(data_dir + imgPath[1:])
                slices = imgs.shape[-1]

                if split == "test":
                    labelPath = None
                else:
                    labelPath = self.jsonData[split][i]["label"]

                row = pd.DataFrame({
                    "imgPath": [data_dir + imgPath[1:]] * slices,
                    "labelPath": [data_dir + labelPath[1:]] * slices if split != 'test' else None,
                    "slice_idx": range(slices)
                })

                sliceinfo = pd.concat([sliceinfo, row], ignore_index=True)
            
            print(f"Total number of slices in {split} set: {sliceinfo.shape[0]}")
            sliceinfo.to_csv(data_dir + f"{split}_sliceinfo.csv", index=False)
            
            return sliceinfo

    def plotRandomSlice(self, sliceinfo : pd.DataFrame, save_plot: bool = False):
        """
        Plot a random slice from the dataframe

        Args:

            sliceinfo (pd.DataFrame): The dataframe containing the slice information
            save_plot (bool): Whether to save the plot or not
        """
        random_index = random.choice(sliceinfo.index)
        random_row = sliceinfo.loc[random_index]

        img = self.loadVolume(random_row["imgPath"], transpose = True)[random_row["slice_idx"]]
        label = self.loadVolume(random_row["labelPath"], transpose = True)[random_row["slice_idx"]]

        _, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img, cmap="gray")
        ax[0].set_title("Image")
        ax[0].axis("off")

        ax[1].imshow(label, cmap="gray")
        ax[1].set_title("Label")
        ax[1].axis("off")
        
        if save_plot:
            plt.savefig(f"slice-{random_index}.png")

        plt.show()


class AMOS_Preprocess:
    def __init__(self, jsonPath: str):
        
        self.dataIngestor = dataIngestion(jsonPath)
        self.label_to_pixel_value: dict = self.dataIngestor.getLabels()


    def getOrganmasks(self, masks: np.ndarray, labels: list[str] = ["liver", "pancreas", "spleen"]) -> np.ndarray:
        """
        Get the masks of the organs from the given masks

        Args:

            masks (np.ndarray): The True masks of the organs
            labels (list[str]): The labels of the organs to be in the masks

        Returns:

            organ_masks (np.ndarray): The masks of the organs with the given labels
        """
        organ_masks = np.zeros_like(masks)

        for i, label in enumerate(labels):
            organ_masks[masks == self.label_to_pixel_value[label]] = i + 1

        return organ_masks
    
    def windowing(self, img: np.ndarray, window: tuple[int, int] = None, window_preset: str = 'ct_abdomen') -> np.ndarray:
        """
        Apply windowing to the image

        Args:

            img (np.ndarray): The image to apply windowing to
            window (tuple[int, int]): The window to apply

        Returns:

            img (np.ndarray): The image with windowing applied
        """
        window_presets = {
            'ct_abdomen': (-135, 215),
            'ct_liver': (-45, 105),
            'ct_spleen': (-135, 215),
            'ct_pancreas': (-135, 215),
        }
        if window is not None:
            return np.clip(img, window[0], window[1])

        if window_preset:
            window = window_presets[window_preset]

        return np.clip(img, window[0], window[1])
    
    def normalize(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize the image

        Args:

            img (np.ndarray): The image to normalize

        Returns:

            img (np.ndarray): The normalized image
        """
        return MinMaxScaler(feature_range = (0, 1)).fit_transform(img).astype(np.float32)
       
    def resize(self, img: np.ndarray, target_shape: tuple[int, int] = (512, 512), label: bool = False) -> np.ndarray:
        """
        Resize the image to the target shape

        Args:

            img (np.ndarray): The image to resize
            target_shape (tuple[int, int]): The target shape to resize the image to

        Returns:

            img (np.ndarray): The resized image
        """
        if label:
            return cv2.resize(img, target_shape, interpolation=cv2.INTER_NEAREST)
        else:
            return cv2.resize(img, target_shape, interpolation=cv2.INTER_LINEAR)
        
    def apply_preprocessing(self, img: np.ndarray,
                             mask: np.ndarray,
                             img_size: tuple[int, int] = (512, 512),
                             labels: list[str] = ["liver", "pancreas", "spleen"],
                             window: tuple[int, int] = None,
                             window_preset: str = 'ct_abdomen'
                            ) -> tuple[torch.tensor, torch.tensor]:
        """
        Apply preprocessing to the image and mask
        
        Args:
            img (np.ndarray): The image to apply preprocessing to
            mask (np.ndarray): The mask to apply preprocessing to
            img_size (tuple[int, int]): The size to resize the image and mask to
            labels (list[str]): The labels of the organs to be in the mask
            window (tuple[int, int]): The window to apply to the image
            window_preset (str): The window preset to apply to the image

        Returns:
            img (torch.tensor): The preprocessed image
            mask (torch.tensor): The preprocessed mask
            """
        # Processing Image
        img = self.windowing(img, window, window_preset)
        img = self.resize(img, img_size)
        img = self.normalize(img)

        # Processing Mask
        mask = self.getOrganmasks(mask, labels = labels)
        mask = self.resize(mask, img_size, label=True)

        return torch.tensor(img).unsqueeze(0).to(torch.float32), torch.tensor(mask).unsqueeze(0).to(torch.float32)
    
class AMOS_Dataset(Dataset):
    def __init__(self, data_dir: str,
                 json_path: str,
                 split: str = "training",
                 img_size: tuple[int, int] = (512, 512),
                 labels: list[str] = ["liver", "pancreas", "spleen"],
                 window: tuple[int, int] = None,
                 window_preset: str = 'ct_abdomen',
                 transform: bool = False,
                 **args):
                 
        self.data_dir = data_dir
        self.dataIngestor = dataIngestion(json_path)
        self.preprocessor = AMOS_Preprocess(json_path)
        self.data = self.dataIngestor.getSliceinfo(data_dir, split)

        self.img_size = img_size
        self.labels = labels
        self.window = window
        self.window_preset = window_preset

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
        ]) if transform else None

        self.label_to_pixel_value = {'background' : 0, **{label: (i + 1) for i, label in enumerate(labels)}}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        img_path, label_path, slice_idx = self.data.iloc[index]

        vol = self.dataIngestor.loadVolume(img_path, transpose=True)
        labels = self.dataIngestor.loadVolume(label_path, transpose=True)

        img = vol[slice_idx]
        mask = labels[slice_idx]

        img, mask = self.preprocessor.apply_preprocessing(img, mask, self.img_size, self.labels, self.window, self.window_preset)

        if self.transform is not None:
            stacked = torch.cat([img, mask], dim=0)
            stacked = self.transform(stacked)
            img, mask = stacked[0:1, :, :], stacked[1:2, :, :]

        return img, mask

if __name__ == "__main__":
    data_dir = sys.argv[1]
    jsonPath = sys.argv[2]

    dataset = AMOS_Dataset(data_dir, jsonPath, split="test")