from .unet import unet_backbone, UNet, UNetTrainer
from .hc_unet import HCUNet, HCUNetTrainer


trainers = {
    'unet' : UNetTrainer,
    'hc_unet': HCUNetTrainer
}