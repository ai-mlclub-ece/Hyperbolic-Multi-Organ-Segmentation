import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

def get_scheduler(optimizer, mode='min', factor=0.1, patience=10):
    """
    Creates a ReduceLROnPlateau scheduler with default parameters
    
    Args:
        optimizer: The optimizer to schedule
        mode: One of 'min' or 'max'. In 'min' mode, lr will be reduced when the quantity monitored has stopped decreasing
        factor: Factor by which the learning rate will be reduced
        patience: Number of epochs with no improvement after which learning rate will be reduced
        
    Returns:
        ReduceLROnPlateau scheduler
    """
    return ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=factor,
        patience=patience
    )

# Example usage:
# scheduler = get_scheduler(optimizer)
# In training loop:
# scheduler.step(val_loss)  # Pass validation loss to scheduler 