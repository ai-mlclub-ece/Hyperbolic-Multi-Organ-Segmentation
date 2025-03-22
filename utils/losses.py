import torch.nn as nn
import torch
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.name = 'cross_entropy'
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x, y):
        """
        x: output of the model
        y: target
        
        return: Cross Entropy loss
        """
        return self.ce(x, y)

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.name = 'dice'

    def dice_coefficient(preds, masks, smooth=1e-6):
        intersection = torch.sum(preds * masks, dim=(1,2,3))
        union = torch.sum(preds, dim=(1,2,3)) + torch.sum(masks, dim=(1,2,3))

        dice_score = (2. * intersection + smooth) / (union + smooth)

        return dice_score.mean()

    def forward(self, x, y):
        """
        x: output of the model
        y: target
        
        return: Dice loss
        """
        return 1 - self.dice_coefficient(x, y)


class HyperUL(nn.Module):
    def __init__(self, c=c,  t=2.718, hr=1.0):
        super(HyperUL, self).__init__()
        self.name = 'hyperul'
        self.c = torch.tensor(c).float()
        self.t = t
        self.hr = hr

    def forward(self, logits, targets):  
        """
        logits: output of the model
        targets: target
        
        return: Hyperbolic uncertainty loss
        """
  
        ce_loss = F.cross_entropy(logits.tensor, targets.squeeze(1).long(), reduction='none')

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
    
class JaccardLoss(nn.Module):
    def __init__(self):
        super(JaccardLoss, self).__init__()
        self.name = 'jaccard'

    def miou(preds, masks, smooth=1e-6):
        intersection = torch.sum(preds * masks, dim=(1,2,3))
        union = torch.sum(preds + masks, dim=(1,2,3)) - intersection

        iou = (intersection + smooth) / (union + smooth)
        
        return iou.mean()
    
    def forward(self, x, y):
        """
        x: output of the model
        y: target
        
        return: Jaccard loss
        """
        return 1 - self.miou(x, y)
    

class hyperbolicdistance(nn.Module):
    def __init__(self, c=c):
        super(hyperbolicdistance, self).__init__()
        self.name = 'hyperbolic_distance'
        pass
    def forward(self, logits, targets):  
        """
        Args:
            logits: output of the model
            targets: target

        Returns:
            Hyperbolic distance
        """
        pass

losses: dict = {
    'cross_entropy': CrossEntropyLoss(),
    'dice': DiceLoss(),
    'jaccard': JaccardLoss(),
    'hyperul': HyperUL(),
    'hyperbolicdistance': hyperbolicdistance()
}

class CombinedLoss(nn.Module):
    def __init__(self, loss_list: list = ['cross_entropy', 'dice'], weights: list[float] = [0.5, 0.5]):
        super(CombinedLoss, self).__init__()
        self.name = 'combined'

        if len(loss_list) != len(weights):
            raise ValueError("Length of loss_list and weights should be equal")
        
        self.losses = [losses[loss] for loss in loss_list]
        
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

losses['combined'] = CombinedLoss()