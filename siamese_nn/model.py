from torch import nn
from torch import optim
import torch.nn.functional as f
from torchvision.models import resnet18


class Net(nn.Module):
    def __init__(self):
        super().__init__()
