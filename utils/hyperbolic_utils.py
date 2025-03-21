import math
import torch


def exp_map_zero(self, x: torch.Tensor, c: float) -> torch.Tensor:
    """
    Projects Euclidean vectors x onto the Poincaré ball using the exponential map at the origin.
    Args:
        x: Tensor of shape (N, D, H, W)
        c: Curvature of the Poincaré ball

        N: Number of samples
        D: Dimension of the input
        H, W: Height and width of the input

    Returns:
        Tensor of same shape with values in the Poincaré ball.
    """
    norm = torch.norm(x, p=2, dim=1, keepdim=True)
    sqrt_c = math.sqrt(c)
    scale = torch.tanh(sqrt_c * norm) / (sqrt_c * (norm + 1e-6))
    return scale * x

def mobius_addition(self, x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    """
    Computes the Mobius addition of two points x and y in the Poincaré ball.
    Args:
        x: Tensor of shape (N, D, H, W)
        y: Tensor of shape (N, D, H, W)
        c: Curvature of the Poincaré ball

        N: Number of samples
        D: Dimension of the input
        H, W: Height and width of the input

    Returns:
        Tensor of same shape with the Mobius addition of x and y.
    """

    # Compute inner product
    inner_prod = torch.sum(x * y, dim=1, keepdim=True)

    # Compute squared norm of x and y
    x_norm_sq = torch.sum(x ** 2, dim=1, keepdim=True)
    y_norm_sq = torch.sum(y ** 2, dim=1, keepdim=True)

    # Compute Mobius norm of x and y
    x_norm = torch.sqrt(torch.clamp(x_norm_sq, min=1e-6))
    y_norm = torch.sqrt(torch.clamp(y_norm_sq, min=1e-6))

    # Compute Mobius norm squared of x and y
    x_norm_sq = torch.clamp(x_norm_sq, min=1e-6)
    y_norm_sq = torch.clamp(y_norm_sq, min=1e-6)

    # Compute Mobius norm squared of x + y
    mobius_norm_sq = 1 + 2 * c * inner_prod + c ** 2 * x_norm_sq * y_norm_sq

    # Compute Mobius addition
    numerator = (1 + 2 * c * inner_prod + c ** 2 * y_norm_sq) * x + (1 - c ** 2 * x_norm_sq) * y
    denominator = 1 + 2 * c * inner_prod + c ** 2 * x_norm_sq * y_norm_sq
    return numerator / denominator

if __name__ == "__main__":
    pass