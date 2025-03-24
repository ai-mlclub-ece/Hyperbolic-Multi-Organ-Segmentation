import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import unet_backbone
from utils import exp_map_zero

from configs import Config, hc_unetConfig
from torch.optim import Adam
from geoopt.optim import RiemannianAdam

class HyperbolicLogisticRegression(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int, c: float, lambda_cp: float):
        super(HyperbolicLogisticRegression, self).__init__()
        """
        Computes per-pixel hyperbolic logits from hyperbolic embeddings.

        Args:
            num_classes(K): int, number of classes
            embedding_dim(D): int, dimension of the hyperbolic embeddings
            c: float, curvature parameter of the hyperbolic space
            lambda_cp: float, scaling factor for the logits
        """

        self.c = c
        self.lambda_cp = lambda_cp

        # Hyperbolic classification parameters (one per class).
        self.p = nn.Parameter(torch.randn(num_classes, embedding_dim))
        self.w = nn.Parameter(torch.randn(num_classes, embedding_dim))

        nn.init.xavier_normal_(self.p)
        nn.init.xavier_normal_(self.w)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute logits from hyperbolic embeddings using the hyperbolic logistic regression model.
        
        Args:
            z: torch.Tensor, hyperbolic embeddings of shape (N, D, H, W)

            N: batch size
            D: embedding dimension
            H,W: Image dimensions
            
        Returns:
            torch.Tensor, logits of shape (N, K, H, W)

            N: batch size
            K: number of classes
        """
        N, D, H, W = z.shape
        K = self.p.shape[0]
        
        # p̂ = -p (reflection of p)
        p_hat = -self.p  # shape: (K, D)
        
        # Reshape for broadcasting
        z_exp = z.unsqueeze(1)              # (N, 1, D, H, W)
        p_hat_exp = p_hat.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1, K, D, 1, 1)
        
        # Compute inner product <p̂, z> for each pixel and class
        dot_pz = (p_hat_exp * z_exp).sum(dim=2)  # (N, K, H, W)
        
        # Compute squared norms
        norm_z_sq = (z_exp ** 2).sum(dim=2)       # (N, 1, H, W)
        norm_p_sq = (p_hat ** 2).sum(dim=1)         # (K,)
        norm_p_sq_exp = norm_p_sq.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1, K, 1, 1)
        
        # Denom for α and β: 1 + 2c <p̂,z> + c² ||p̂||² ||z||²
        denom = 1 + 2 * self.c * dot_pz + (self.c ** 2) * norm_p_sq_exp * norm_z_sq  # (N, K, H, W)
        
        # Compute coefficients α and β (reformulation, Equation 9)
        alpha = (1 + 2 * self.c * dot_pz + self.c * norm_z_sq) / (denom + 1e-6)  # (N, K, H, W)
        beta  = (1 - self.c * norm_p_sq_exp) / (denom + 1e-6)               # (N, K, H, W)
        
        # Compute <p̂, w> for each class
        dot_pw = (p_hat * self.w).sum(dim=1)           # (K,)
        dot_pw_exp = dot_pw.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1, K, 1, 1)
        
        # Compute <z, w> for each pixel and class
        w_exp = self.w.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1, K, D, 1, 1)
        dot_zw = (z_exp * w_exp).sum(dim=2)  # (N, K, H, W)
        
        # Combined inner product term: α <p̂, w> + β <z, w>
        inner_prod = alpha * dot_pw_exp + beta * dot_zw  # (N, K, H, W)
        
        # Compute squared norm of (p̂ ⊕ z): α²||p̂||² + 2αβ<p̂, z> + β²||z||²
        mobius_norm_sq = (alpha ** 2) * norm_p_sq_exp + 2 * alpha * beta * dot_pz + (beta ** 2) * norm_z_sq  # (N, K, H, W)
        
        # Norm of orientation parameters w
        w_norm = self.w.norm(p=2, dim=1)  # (K,)
        w_norm_exp = w_norm.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1, K, 1, 1)
        
        # Argument for asinh: (2*sqrt(c)*inner_prod) / ((1 - c * mobius_norm_sq) * w_norm)
        numerator = 2 * math.sqrt(self.c) * inner_prod
        denominator = (1 - self.c * mobius_norm_sq).clamp(min=1e-6) * (w_norm_exp.clamp(min=1e-6))
        asinh_arg = numerator / (denominator + 1e-6)
        
        # Compute logit: scaled by lambda_cp/(w_norm*sqrt(c))
        logit = (self.lambda_cp / (w_norm_exp * math.sqrt(self.c))) * torch.asinh(asinh_arg)
        return logit
    

class HCUNet(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int, curvature: float, lambda_cp: float):
        super(HCUNet, self).__init__()
        """
        Hyperbolic Classifier UNet with Hyperbolic Logistic Regression head.
        
        Args:
            num_classes: int, number of classes
            embedding_dim: int, dimension of the hyperbolic embeddings
            curvature: float, curvature parameter of the hyperbolic space
            lambda_cp: float, scaling factor for the logits   
        """
        self.unet = unet_backbone(out_channels = embedding_dim)
        self.unet.apply(self.unet.init_weights)

        self.classifier = HyperbolicLogisticRegression(num_classes,
                                                       embedding_dim,
                                                       curvature,
                                                       lambda_cp)
        
        self.activation = nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model
        
        Args:
        
            x: torch.Tensor, input tensor of shape (N, C, H, W)
            
            N: batch size
            C: number of channels
            H, W: image dimensions
        
        Returns:
        
            torch.Tensor, logits of shape (N, K, H, W)
            
            N: batch size
            K: number of classes
            H, W: image dimensions
        """
        # Compute UNet embeddings
        embedding = self.unet(x)

        # Compute hyperbolic embeddings
        hyperbolic_embedding = exp_map_zero(embedding, self.hc_logreg.c)
        
        # Compute logits with hyperbolic logistic regression
        logits = self.classifier(hyperbolic_embedding)

        return self.activation(logits)
    
class HCUNetTrainer:
    def __init__(self, config: Config = hc_unetConfig()):
        self.model = HCUNet(num_classes = len(config.labels),
                            embedding_dim = config.embedding_dim,
                            curvature = config.curvature,
                            lambda_cp = config.lambda_cp)
        
        self.optimizers = [Adam(self.model.unet.parameters(), lr = config.learning_rate),
                           RiemannianAdam(self.model.classifier.parameters(), lr = config.learning_rate)]
        
if __name__ == "__main__":
    model = HCUNet(num_classes=3,
                   embedding_dim=256,
                   curvature=0.1,
                   lambda_cp=1.0
    )
        