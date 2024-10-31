import abc
import torch.nn as nn


class BaseModel(abc.ABC):
    def __init__(self, black_box: nn.Module):
        self.net = black_box

    def forward(self, img):
        pass
