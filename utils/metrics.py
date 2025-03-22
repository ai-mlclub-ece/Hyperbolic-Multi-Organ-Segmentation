import torch

class baseMetric:
    def __init__(self):
        pass

    def metric(preds, masks):
        pass

    def compute(self, preds, masks, labels_to_pixels: dict):
        scores = {}
        for label, pixel_value in labels_to_pixels.items():
            pred = preds[preds == pixel_value].float()
            mask = masks[masks == pixel_value].float()

            scores[label] =  self.metric(pred, mask)

        return scores, sum(scores.values())/len(labels_to_pixels)

class dicescore(baseMetric):
    def __init__(self):
        
        self.name = 'dice_score'

    def metric(preds, masks, smooth=1e-6):
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
    

class miou(baseMetric):
    def __init__(self):
        
        self.name = 'miou'

    def metric(preds, targets):
        """
        Evaluates the mean Intersection over Union (mIoU) for the predicted and target masks

        Args:
            pred: predicted masks
            target: target masks
        
        Returns:
            mIoU: mean Intersection over Union (mIoU)
        """
    
        targets = targets.float()

        intersection = (preds * targets).sum(dim=(1,2,3))
        union = (preds + targets).sum(dim=(1,2,3)) - intersection 

        iou = (intersection + 1e-8) / (union + 1e-8)
        return iou.mean().item()
    
class precision(baseMetric):
    def __init__(self):
        
        self.name = 'precision'

    def metric(preds, targets):
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
    def __init__(self):
        
        self.name = 'recall'
        
    def metric(preds, targets):
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