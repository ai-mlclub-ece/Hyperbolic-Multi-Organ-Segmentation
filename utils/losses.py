import torch.nn as nn
import torch
import torch.nn.functional as F

class bceloss(nn.Module):
    def __init__(self):
        super(bceloss, self).__init__()
        self.name = 'bce'
        self.bce = nn.BCELoss()
        
    def forward(self, x, y):
        """
        x: output of the model
        y: target
        
        return: BCE loss
        """
        return self.bce(x, y)

class diceloss(nn.Module):
    def __init__(self):
        super(diceloss, self).__init__()
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
    
class jaccardloss(nn.Module):
    def __init__(self):
        super(jaccardloss, self).__init__()
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
        pass
      
        