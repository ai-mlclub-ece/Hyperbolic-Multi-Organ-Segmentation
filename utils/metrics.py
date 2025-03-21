import torch.nn as nn
import torch

class dicescore(nn.Module):
    def __init__(self):
        super(dicescore, self).__init__()

    def dice_coefficient(preds, masks, smooth=1e-6):
        """
        Evaluates the dice coefficient for the predicted and target masks
        
        Args:
            preds: predicted masks
            masks: target masks
            smooth: smoothing factor to avoid division by zero

        Returns:
            dice_score: dice coefficient
        """
        intersection = torch.sum(preds * masks, dim=(1,2,3))
        union = torch.sum(preds, dim=(1,2,3)) + torch.sum(masks, dim=(1,2,3))

        dice_score = (2. * intersection + smooth) / (union + smooth)

        return dice_score.mean().item() 

class miou(nn.Module):
    def __init__(self):
        super(miou, self).__init__()
    def mIou(pred, target):
        """
        Evaluates the mean Intersection over Union (mIoU) for the predicted and target masks

        Args:
            pred: predicted masks
            target: target masks
        
        Returns:
            mIoU: mean Intersection over Union (mIoU)
        """
    
        target = target.float()

        intersection = (pred * target).sum(dim=(1,2,3))
        union = (pred + target).sum(dim=(1,2,3)) - intersection 

        iou = (intersection) / (union + 1e-8)
        return iou.mean().item()
    
class precision(nn.Module):
    def __init__(self):
        super(precision, self).__init__()
    def precision(preds, targets):
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
    
class recall(nn.Module):
    def __init__(self):
        super(recall,self).__init__()
    def recall(preds, targets):
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