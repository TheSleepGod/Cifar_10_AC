import torch
import torch.nn as nn
import torch.nn.functional as f

from .base_model import BaseModel


class JacobianModel(BaseModel):
    def __init__(self, black_box: nn.Module, attack_rate: float):
        super().__init__(black_box)
        self.net = black_box
        self.attack_rate = attack_rate

    def __call__(self, imgs, labels):
        jacobians = torch.autograd.functional.jacobian(func=self.net.__call__(), inputs=imgs)
        # print(jacobians.size())
        jacobians = f.normalize(jacobians)
        for i in range(imgs.size()[0]):
            imgs[i] -= self.attack_rate * torch.sign(jacobians[i][torch.argmax(labels[i])].squeeze())

        return imgs
