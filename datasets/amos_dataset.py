import os
import sys
import cv2
import json
import random
from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

from torch.utils.data import Dataset

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
        label_to_pixel= {label: pixel_value for pixel_value, label in pixel_to_label.items()}
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

    
    def getSliceinfo(self, data_dir: str, dataframe_dir: str, split: str = "training") -> pd.DataFrame:
        """
        Get the slice information of the given split of the dataset

        Args:

            data_dir (str): The path to the dataset directory
            split (str): The split of the dataset to load. Can be "training", "validation" or "test"
            dataframe_dir (str): The path to the directory where the dataframes are stored or to be stored

        Returns:

            sliceinfo (pd.DataFrame): A dataframe containing the slice information of the dataset
        """
        if os.path.exists(dataframe_dir + f"{split}_sliceinfo.csv"):
            sliceinfo = pd.read_csv(dataframe_dir + f"{split}_sliceinfo.csv")
            print(f"Loaded {split} set from {dataframe_dir}...")
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
                    "labelPath": [data_dir + labelPath[1:]] * slices,
                    "slice_idx": range(slices)
                })

                sliceinfo = pd.concat([sliceinfo, row], ignore_index=True)
            
            print(f"Total number of slices in {split} set: {sliceinfo.shape[0]}")
            sliceinfo.to_csv(dataframe_dir + f"{split}_sliceinfo.csv", index=False)
            
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

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
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
    def __init__(self, data_dir: str, jsonPath: str):
        self.data_dir = data_dir
        self.jsonPath = jsonPath
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

        for label in labels:
            organ_masks[masks == self.label_to_pixel_value[label]] = self.label_to_pixel_value[label]

        return organ_masks
    
    def windowing(self, img: np.ndarray, window: tuple[int, int], window_preset: str = 'ct_abdomen') -> np.ndarray:
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
        if window:
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
        return (img - img.min()) / (img.max() - img.min())
    
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
    
class AMOS_Dataset(Dataset):
    pass