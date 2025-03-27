import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class inferVisualizer:
    def __init__(self, criterion):
        self.criterion = criterion

    def visualize(self, image: np.ndarray, mask: np.ndarray,
                        pred: np.ndarray, save_path: str = None):
        
        fig, ax = plt.subplots(1, 4, figsize=(15, 5))

        ax[0].imshow(image, cmap='gray')
        ax[0].set_title('Image')
        ax[0].axis('off')

        ax[1].imshow(mask, cmap='gray')
        ax[1].set_title('Mask')
        ax[1].axis('off')

        ax[2].imshow(pred, cmap='gray')
        ax[2].set_title('Prediction')
        ax[2].axis('off')

        diff = mask != pred
        hot_mask = mask.copy()
        hot_mask[diff] = 255
        
        ax[3].imshow(hot_mask, cmap='hot')
        ax[3].set_title(f'difference}')
        ax[3].axis('off')

        if save_path is not None:
            plt.savefig(save_path)
        plt.close()
    
    def visualize_batch(self, images: torch.Tensor, masks: torch.Tensor,
                            preds: torch.Tensor, save_path: str):
        
        images = images.cpu().numpy()
        masks = masks.cpu().numpy()
        preds = preds.cpu().numpy()

        for i in range(images.shape[0]):
            save_path = save_path.replace('.png', f'_{i}.png')
            self.visualize(images[i, 0], masks[i, 0], preds[i, 0], save_path=save_path)

class trainLogVisualizer:
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.logs = pd.read_csv(log_path)

    def visualize(self, save_path: str = None):
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))

        ax[0].plot(self.logs['epoch'], self.logs['train_loss'], label='train_loss')
        ax[0].plot(self.logs['epoch'], self.logs['val_loss'], label='val_loss')
        ax[0].set_title('Loss')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].legend()

        ax[1].plot(self.logs['epoch'], self.logs['train_dice'], label='train_dice')
        ax[1].plot(self.logs['epoch'], self.logs['val_dice'], label='val_dice')
        ax[1].set_title('Dice')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Dice')
        ax[1].legend()

        if save_path is not None:
            plt.savefig(save_path)
        plt.close()
