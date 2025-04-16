import torch
import matplotlib.pyplot as plt

class baseMetric:
    def __init__(self, labels, labels_to_pixels):
        self.labels = labels
        self.labels_to_pixels = labels_to_pixels

    def metric(self, preds, masks):
        pass

    def compute(self, preds, masks):
        scores = {}
        
        for label in self.labels:

            pred = (preds == self.labels_to_pixels[label]).float()
            mask = (masks == self.labels_to_pixels[label]).float()
            
            scores[label] =  self.metric(pred, mask)

        return scores, sum(scores.values())/len(self.labels)

class dicescore(baseMetric):
    def __init__(self, labels, labels_to_pixels):
        
        self.name = 'dice_score'

        self.labels = labels
        self.labels_to_pixels = labels_to_pixels

    def metric(self, preds, masks, smooth=1e-6):
        """
        Evaluates the dice coefficient for the predicted and target masks
        
        Args:
            preds: predicted masks
            masks: target masks
            smooth: smoothing factor to avoid division by zero

        Returns:
            dice_score: dice coefficient
        """

        intersection = torch.sum(preds * masks)
        union = torch.sum(preds) + torch.sum(masks)

        dice_score = (2. * intersection + smooth) / (union + smooth)

        return dice_score.mean().item()
    

class miou(baseMetric):
    def __init__(self, labels, labels_to_pixels):
        
        self.name = 'miou'

        self.labels = labels
        self.labels_to_pixels = labels_to_pixels

    def metric(self, preds, targets):
        """
        Evaluates the mean Intersection over Union (mIoU) for the predicted and target masks

        Args:
            pred: predicted masks
            target: target masks
        
        Returns:
            mIoU: mean Intersection over Union (mIoU)
        """
    
        targets = targets.float()

        intersection = (preds * targets).sum()
        union = (preds + targets).sum() - intersection 

        iou = (intersection + 1e-8) / (union + 1e-8)
        return iou.mean().item()
    
class precision(baseMetric):
    def __init__(self, labels, labels_to_pixels):
        
        self.name = 'precision'

        self.labels = labels
        self.labels_to_pixels = labels_to_pixels

    def metric(self, preds, targets):
        """
        Evaluates the precision for the predicted and target masks

        Args:
            preds: predicted masks
            targets: target masks

        Returns:
            precision: precision
        """

        targets = targets.float()

        tp = (preds * targets).sum()
        fp = (preds * (1 - targets)).sum()

        precision = tp / (tp + fp + 1e-8)
        return precision.item()
    
class recall(baseMetric):
    def __init__(self, labels, labels_to_pixels):
        
        self.name = 'recall'

        self.labels = labels
        self.labels_to_pixels = labels_to_pixels
        
    def metric(self, preds, targets):
        """
        Evaluates the recall for the predicted and target masks

        Args:
            preds: predicted masks
            targets: target masks

        Returns:
            recall: recall
        """
        targets = targets.float()

        tp = (preds * targets).sum()
        fn = ((1 - preds) * targets).sum()

        recall = tp/ (tp + fn + 1e-8)
        return recall.item()