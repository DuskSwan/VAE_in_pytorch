# encoding: utf-8

import argparse
import os
import sys
from os import mkdir
import time

import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage

sys.path.append('.')
from config import cfg
from data import make_data_loader
from modeling import build_model
from solver import make_optimizer
from utils import set_random_seed


class loss_fn:
    def __init__(self, kl_weight=0.00025):
        self.kl_weight = kl_weight

    def __call__(self, y, y_hat, mean, logvar):
        self.recons_loss = F.mse_loss(y_hat, y)
        self.kl_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar), 1), 0)
        self.loss = self.recons_loss + self.kl_loss * self.kl_weight
        return self.loss


def train_without_engine(cfg, dataloader, model):
    device = cfg.DEVICE
    lr = cfg.SOLVER.BASE_LR
    n_epochs = cfg.SOLVER.MAX_EPOCHS

    optimizer = torch.optim.Adam(model.parameters(), lr)
    dataset_len = len(dataloader.dataset)
    loss_f = loss_fn(kl_weight=cfg.SOLVER.KL_WEIGHT)

    begin_time = time.time()

    # train
    for i in range(n_epochs):
        loss_sum = 0
        for x in dataloader:
            x = x.to(device)
            y_hat, mean, logvar = model(x)
            loss = loss_f(x, y_hat, mean, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss
        loss_sum /= dataset_len
        training_time = time.time() - begin_time
        minute = int(training_time // 60)
        second = int(training_time % 60)
        print(f'epoch {i}: loss {loss_sum} {minute}:{second}')
        torch.save(model, 'output/VAEmodel.pth')

def reconstruct(cfg, dataloader, model, n=1):
    device = cfg.DEVICE
    model.eval()
    batch = next(iter(dataloader))
    assert n <= batch.size(0)
    x = batch[0:n, ...].to(device) # n, 3, 64, 64
    for i in range(n):
        input = x[i:i+1, ...]
        output = model(input)[0]
        output = output[0].detach().cpu()
        raw = batch[i].detach().cpu()
        combined = torch.cat((output, raw), 1)
        img = ToPILImage()(combined)
        img.save(f'output/reconstruct{i}.jpg')

def generate(cfg, model, n=1):
    device = cfg.DEVICE
    model.eval()

    for i in range(n):
        output = model.sample(device)
        output = output[0].detach().cpu()
        img = ToPILImage()(output)
        img.save(f'output/generate{i}.jpg')

def main():
    set_random_seed(cfg.SEED)
    model = build_model(cfg)
    train_loader = make_data_loader(cfg, is_train=True)
    train_without_engine(cfg, train_loader, model)
    # model = torch.load('output/VAEmodel.pth')

    val_loader = make_data_loader(cfg, is_train=False)
    reconstruct(cfg, val_loader, model, 4)

    generate(cfg, model, 4)


if __name__ == '__main__':
    main()
