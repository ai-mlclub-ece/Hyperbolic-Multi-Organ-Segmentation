import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import Config, unetConfig
from torch.optim import Adam

class unet_backbone(nn.Module):
    def __init__(self, out_channels: int):
        super(unet_backbone, self).__init__()
        """
        Baseline UNet Backbone

        Args:
            out_channels: int, number of output channels
        """
        self.e11 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)


        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1) 

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.outconv = nn.Conv2d(64, out_channels, kernel_size=1)

    def init_weights(self, m: nn.Module):
        """
        Initialize weights of the model
        
        Args:
            m: nn.Module, model to initialize
        """
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model
        
        Args:
            x: torch.Tensor, input tensor
        
        Returns:
            torch.Tensor, output tensor
        """

        # Encoder
        x = F.relu(self.e11(x))
        x1 = F.relu(self.e12(x))
        x = self.pool1(x1)

        x = F.relu(self.e21(x))
        x2 = F.relu(self.e22(x))
        x = self.pool2(x2)

        x = F.relu(self.e31(x))
        x3 = F.relu(self.e32(x))
        x = self.pool3(x3)

        x = F.relu(self.e41(x))
        x4 = F.relu(self.e42(x))
        x = self.pool4(x4)

        x = F.relu(self.e51(x))
        x = F.relu(self.e52(x))

        # Decoder
        x = self.upconv1(x)
        x = torch.cat([x,x4],dim=1)
        x = F.relu(self.d11(x))
        x = F.relu(self.d12(x))

        x = self.upconv2(x)
        x = torch.cat([x,x3],dim=1)
        x = F.relu(self.d21(x))
        x = F.relu(self.d22(x))

        x = self.upconv3(x)
        x = torch.cat([x,x2],dim=1)
        x = F.relu(self.d31(x))
        x = F.relu(self.d32(x))

        x = self.upconv4(x)
        x = torch.cat([x,x1],dim=1)
        x = F.relu(self.d41(x))
        x = F.relu(self.d42(x))

        return self.outconv(x)

class UNet(nn.Module):
    def __init__(self, num_classes: int):
        super(UNet, self).__init__()
        """
        UNet model with activation function at the end
        
        Args:
            out_channels: int, number of output channels
        """
        self.unet_backbone = unet_backbone(num_classes)
        self.unet_backbone.apply(self.unet_backbone.init_weights)

        if num_classes == 1:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model

        Args:
            x: torch.Tensor, input tensor

        Returns:
            torch.Tensor, output tensor
        """
        x = self.unet_backbone(x)
        return self.final_activation(x), x
    
class UNetTrainer:
    def __init__(self, config: Config = unetConfig()):
        self.model = UNet(num_classes = len(config.labels) + 1)
        self.optimizers = [Adam(self.model.parameters(), lr = config.learning_rate)]
        
if __name__ == "__main__":
    model = UNet(num_classes=3)