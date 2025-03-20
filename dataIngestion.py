import os
import sys
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
import nibabel as nib

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
        """
        return self.jsonData["labels"]
    
    def loadVolume(self, path:str, transpose: bool = False) -> np.ndarray:
        """
        Load the volume file (.nii.gz) from the given path
        """

        if not transpose:
            return nib.load(path).get_fdata()
        else:
            return nib.load(path).get_fdata().transpose(2, 1, 0)

    
    def getSliceinfo(self, root_dir: str, split: str = "training") -> pd.DataFrame:
        """
        Get the slice information of the given split of the dataset
        """

        print(f"Loading {split} set...")

        sliceinfo = pd.DataFrame(columns=["imgPath", "labelPath", "slice_idx"])

        for i in tqdm(range(len(self.jsonData[split]))):

            imgPath = self.jsonData[split][i]["image"]
            imgs = self.loadVolume(root_dir + imgPath[1:])
            slices = imgs.shape[-1]

            if split == "test":
                labelPath = None
            else:
                labelPath = self.jsonData[split][i]["label"]

            row = pd.DataFrame({
                "imgPath": [imgPath] * slices,
                "labelPath": [labelPath] * slices,
                "slice_idx": range(slices)
            })

            sliceinfo = pd.concat([sliceinfo, row], ignore_index=True)
        
        print(f"Total number of slices in {split} set: {sliceinfo.shape[0]}")
        return sliceinfo



if __name__ == "__main__":
    di = dataIngestion(sys.argv[1])