import torch
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
        ax[3].set_title(f'diff = {self.criterion(mask, pred)}')
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

