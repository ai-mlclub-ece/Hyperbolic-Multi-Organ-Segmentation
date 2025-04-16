import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class CrossEntropyLoss(nn.Module):
    def __init__(self, labels : list, labels_to_pixels: dict, class_weights: list = None):
        super(CrossEntropyLoss, self).__init__()
        self.name = 'cross_entropy'

        self.labels = labels
        self.labels_to_pixels = labels_to_pixels
        
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights).float()
        else:
            self.class_weights = None

        self.ce = nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(self, x, y):
        """
        x: raw logits from model
        y: target
        
        return: Cross Entropy loss
        """
        return self.ce(x, y.squeeze(1).long())

class DiceLoss(nn.Module):
    def __init__(self, labels : list, labels_to_pixels: dict, class_weights: list = None):
        super(DiceLoss, self).__init__()
        self.name = 'dice'

        self.labels = labels
        self.labels_to_pixels = labels_to_pixels
        
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights).float()
        else:
            self.class_weights = torch.ones(len(labels))

    def forward(self, preds, masks):
        """
        x: raw logits from model
        y: target
        
        return: Dice loss
        """
        # Apply softmax to logits
        preds = F.softmax(preds, dim=1)
        
        losses = []
        masks = masks.squeeze(1)

        for i, label in enumerate(self.labels):
            pred = preds[:, i, :, :].float()
            mask = (masks == self.labels_to_pixels[label]).float()
            weight = self.class_weights[i] if i < len(self.class_weights) else 1.0
            losses.append(self.dice_coefficient(pred, mask) * weight)

        return 1 - torch.mean(torch.stack(losses))

    def dice_coefficient(self, preds, masks, smooth=1e-6):
        intersection = torch.sum(preds * masks)
        union = torch.sum(preds) + torch.sum(masks)

        dice_score = (2. * intersection + smooth) / (union + smooth)

        return dice_score.mean()
    
class JaccardLoss(nn.Module):
    def __init__(self, labels : list, labels_to_pixels: dict, class_weights: list = None):
        super(JaccardLoss, self).__init__()
        self.name = 'jaccard'

        self.labels = labels
        self.labels_to_pixels = labels_to_pixels
        
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights).float()
        else:
            self.class_weights = torch.ones(len(labels))

    def miou(self, preds, masks, smooth=1e-6):
        intersection = torch.sum(preds * masks)
        union = torch.sum(preds + masks) - intersection

        iou = (intersection + smooth) / (union + smooth)
        
        return iou.mean()
    
    def forward(self, preds, masks):
        """
        x: raw logits from model
        y: target
        
        return: Jaccard loss
        """
        # Apply softmax to logits
        preds = F.softmax(preds, dim=1)
        
        losses = []
        masks = masks.squeeze(1)

        for i, label in enumerate(self.labels_to_pixels):
            pred = preds[:, i, :, :].float()
            mask = (masks == self.labels_to_pixels[label]).float()
            losses.append(self.miou(pred, mask) * self.class_weights[i])

        return 1 - torch.mean(torch.stack(losses))
    
class HyperUL(nn.Module):
    def __init__(self, labels : list, labels_to_pixels: dict, c= 0.1, t=2.718, hr=1.0, class_weights: list = None):
        super(HyperUL, self).__init__()
        self.name = 'hyperul'
        self.c = torch.tensor(c).float()
        self.t = t
        self.hr = hr

        self.labels = labels
        self.labels_to_pixels = labels_to_pixels
        
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights).float()
        else:
            self.class_weights = None

    def forward(self, logits, targets):  
        """
        logits: raw logits from model
        targets: target
        
        return: Hyperbolic uncertainty loss
        """
        ce_loss = F.cross_entropy(logits, targets.squeeze(1).long(), weight=self.class_weights, reduction='none')

        sqrt_c = torch.sqrt(self.c)
        norm  = torch.norm(logits.tensor,dim=1)
        hyperbolic_distances = 2 * sqrt_c * torch.atanh(sqrt_c * torch.clamp(norm, max = 1 - 1e-6))

        max_distance = hyperbolic_distances.max()
        scaled_distances = hyperbolic_distances / (max_distance + 1e-8)

        uncertainty_weights = 1 / torch.log(self.t + scaled_distances + 1e-8)

        threshold = torch.quantile(hyperbolic_distances, self.hr)
        mask = hyperbolic_distances <= threshold

        if mask.sum() > 0:
            hyperul = (uncertainty_weights * ce_loss)[mask].mean()
        else:
            hyperul = (uncertainty_weights * ce_loss).mean()
        return hyperul

class hyperbolicdistance(nn.Module):
    def __init__(self, labels : list, labels_to_pixels: dict, c=0.1, class_weights: list = None):
        super(hyperbolicdistance, self).__init__()
        self.name = 'hyperbolic_distance'
        self.c = torch.tensor(c).float()

        self.labels = labels
        self.labels_to_pixels = labels_to_pixels
        
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights).float()
        else:
            self.class_weights = torch.ones(len(labels))

    def forward(self, logits, targets):  
        """
        Args:
            logits: output of the model
            targets: target

        Returns:
            Hyperbolic distance
        """
        pass

criterions: dict = {
    'cross_entropy': CrossEntropyLoss,
    'dice': DiceLoss,
    'jaccard': JaccardLoss,
    'hyperul': HyperUL,
    'hyperbolicdistance': hyperbolicdistance
}

class CombinedLoss(nn.Module):
    def __init__(self, labels : list, labels_to_pixels: dict, loss_list: list = ['cross_entropy', 'dice'],
                 weights: list[float] = [0.5, 0.5], class_weights: list = None):
        super(CombinedLoss, self).__init__()
        self.name = 'combined'

        self.labels = labels
        self.labels_to_pixels = labels_to_pixels

        if len(loss_list) != len(weights):
            raise ValueError("Length of loss_list and weights should be equal")
        
        self.losses = [criterions[loss](labels=labels, labels_to_pixels=labels_to_pixels, class_weights=class_weights) for loss in loss_list]
        
        self.weights = weights
    
    def forward(self, x, y):
        """
        x: output of the model
        y: target
        
        return: Combined loss
        """
        loss = 0
        for i, loss_fn in enumerate(self.losses):
            loss += self.weights[i] * loss_fn(x, y)
        return loss

criterions['combined'] = CombinedLoss