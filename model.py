import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocess import get_window_samples
from model_parts import init_weights

# Use manual seed to reproduce results
#torch.manual_seed(1)

# Use cuda if it's allowed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_model(init_filters=64, n_classes=1):
    """ Use this function to get the model. """
    return UNet(init_filters, n_classes).to(device)


class UNet(nn.Module):
    def __init__(self, init_filters=64, n_classes=1):
        super(UNet, self).__init__()
        self.conv_1 = UNetDoubleConv(1, init_filters)
        self.down_2 = UNetDownLayer(init_filters, init_filters*2)
        self.down_3 = UNetDownLayer(init_filters*2, init_filters*4)
        self.down_4 = UNetDownLayer(init_filters*4, init_filters*8)
        self.down_5 = UNetDownLayer(init_filters*8, init_filters*16)
        self.up_6 = UNetUpLayer(init_filters*16, init_filters*8)
        self.up_7 = UNetUpLayer(init_filters*8, init_filters*4)
        self.up_8 = UNetUpLayer(init_filters*4, init_filters*2)
        self.up_9 = UNetUpLayer(init_filters*2, init_filters)
        self.out = UNetOutLayer(init_filters, n_classes)
        
    def forward(self, x):
        x1 = self.conv_1(x)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)
        x = self.down_5(x4)
        x = self.up_6(x, x4)
        x = self.up_7(x, x3)
        x = self.up_8(x, x2)
        x = self.up_9(x, x1)
        x = self.out(x)
        return x
    
    def predict(self, x):
        x_samples = torch.tensor(get_window_samples(x, 80)).unsqueeze_(1).to(device)
        x = (x_samples - self.forward(x_samples)).cpu().squeeze_(1).view(-1, 80)[:x.shape[0]]
        return x


# Two Convolution Layers with BN, Dropout and Activation
class UNetDoubleConv(nn.Module):
    def __init__(self, input_filters, out_filters):
        super(UNetDoubleConv, self).__init__()
        
        self.sequence = nn.Sequential(
            nn.Conv2d(input_filters, out_filters, (3, 3), padding=1, bias=False),
            nn.BatchNorm2d(out_filters),
            nn.Dropout2d(p=0.1),
            Mish(),
            nn.Conv2d(out_filters, out_filters, (3, 3), padding=1, bias=False),
            nn.BatchNorm2d(out_filters),
            Mish()
        )
        
        self.sequence.apply(init_weights)
        
    def forward(self, x):
        x = self.sequence(x)
        return x


# Downsample Layer
class UNetDownLayer(nn.Module):
    def __init__(self, input_filters, out_filters):
        super(UNetDownLayer, self).__init__()
        
        self.sequence = nn.Sequential(
            nn.MaxPool2d((2, 2), stride=2),
            UNetDoubleConv(input_filters, out_filters)
        )
        
        self.sequence.apply(init_weights)
    
    def forward(self, x):
        x = self.sequence(x)
        return x


# Upsample Layer
class UNetUpLayer(nn.Module):
    def __init__(self, input_filters, out_filters):
        super(UNetUpLayer, self).__init__()
        
        self.up = nn.Sequential(
            nn.ConvTranspose2d(input_filters, input_filters//2, (2, 2), stride=2, bias=False),
            nn.BatchNorm2d(input_filters//2),
            Mish()
        )
        self.conv = UNetDoubleConv(input_filters, out_filters)
        
        self.up.apply(init_weights)
    
    def forward(self, x1, x2):
        x = self.up(x1)
        x = torch.cat((x2, x), dim=1)
        x = self.conv(x)
        return x


# Output Layer
class UNetOutLayer(nn.Module):
    def __init__(self, input_filters, n_classes=1):
        super(UNetOutLayer, self).__init__()
        
        self.sequence = nn.Sequential(
            nn.Conv2d(input_filters, n_classes, (1, 1)),
            #nn.Tanh()
        )
        
        self.sequence.apply(init_weights)
        
    def forward(self, x):
        x = self.sequence(x)
        return x


# Activation Function
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))
