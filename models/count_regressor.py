import torch
from torch.nn import Module
import torchvision.models as tmodels
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb

M_EPS = 1e-16


class Trainer(object):
    def __init__(self, device, lr, max_epoch, val_epoch, **kwargs):
        self.device = device
        self.lr = lr
        self.max_epoch = max_epoch
        self.val_epoch = val_epoch

    def setup(self, backbone):
        self.device = torch.device(self.device)

        self.backbone = backbone.to(self.device)

        self.optimizer = optim.Adam(self.backbone.parameters(), lr=self.lr)

        self.start_epoch = 0

        self.loss = nn.MSELoss().to(self.device)

    def train_step(self, inputs, gt_discrete, logger):
        inputs = inputs.to(self.device)
        valid = inputs[:, [-1,]].to(self.device)

        target_count = gt_discrete.sum(dim=(1, 2, 3)).to(self.device)

        with torch.set_grad_enabled(True):
            outputs = (self.backbone(inputs) * valid).sum(dim=(1, 2, 3))

            loss = self.loss(outputs, target_count)

            logger.log({
                'train/loss': loss.item(),
            })

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict(self, inputs):
        inputs = inputs.to(self.device)
        valid = inputs[:, [-1,]].to(self.device)
        with torch.no_grad():
            outputs = self.backbone(inputs) * valid
        return outputs

    def train(self):
        self.backbone.train()

    def eval(self):
        self.backbone.eval()

if __name__ == "__main__":
    pass