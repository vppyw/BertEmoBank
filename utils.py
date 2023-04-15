import os
import json
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch import optim

def prepare_env(args):
    os.makedirs(args.dir, exist_ok=args.overwrite)
    args.config_file = os.path.join(args.dir, 'config.json')
    args.log_file = os.path.join(args.dir, 'log.json')
    args.model_file = os.path.join(args.dir, 'model.pt')
    args.token_file = os.path.join(args.dir, 'model.pt')

    with open(args.config_file, 'w') as f:
        f.write(json.dumps(vars(args), indent=2))

    with open(args.log_file, 'w') as f:
        f.write(json.dumps([]))
    
def same_seed(seed):
    random.seed(seed)

def get_split_loader(
    dataset='emobank',
    batch_size=16,
    num_workers=4
):
    def valid_row(row):
        return isinstance(row['V'], float) and \
               isinstance(row['A'], float) and \
               isinstance(row['D'], float) and \
               isinstance(row['text'], str)
    if dataset == 'emobank':
        raw_datas = pd.read_csv("./EmoBank/corpus/emobank.csv")
        train_datas = []
        valid_datas = []
        test_datas = []
        for idx, row in raw_datas.iterrows():
            if not valid_row(row):
                continue
            if row['split'] == 'train':
                train_datas.append([row['V'], row['A'], row['D'], row['text']])
            elif row['split'] == 'dev':
                valid_datas.append([row['V'], row['A'], row['D'], row['text']])
            elif row['split'] == 'test':
                test_datas.append([row['V'], row['A'], row['D'], row['text']])
        train_loader = DataLoader(
                        train_datas,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                       )
        valid_loader = DataLoader(
                        valid_datas,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                       )
        test_loader = DataLoader(
                        test_datas,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                      )
        return train_loader, valid_loader, test_loader
    else:
        raise NotImplementedError

def get_loss_fn(args):
    return nn.MSELoss()

def get_optimizer(args, model):
    return optim.AdamW(model.parameters(), lr=args.lr)

def step(args, model, loss_fn, batch):
    preds = model(batch[3], device=args.device)
    v_loss = loss_fn(
                batch[0].float().to(args.device),
                preds[:,0].squeeze()
             )
    a_loss = loss_fn(
                batch[1].float().to(args.device),
                preds[:,1].squeeze()
             )
    d_loss = loss_fn(
                batch[2].float().to(args.device),
                preds[:,2].squeeze()
             )
    loss = v_loss + a_loss + d_loss
    return {
            'loss': loss,
            'v_loss': v_loss,
            'a_loss': a_loss,
            'd_loss': d_loss
           }

def train(args, model, optim, loss_fn, loader):
    loss_name = ['loss', 'v_loss', 'a_loss', 'd_loss']
    logs = {l:[] for l in loss_name}
    model = model.to(args.device)
    model.train()
    for batch in loader:
        log = step(args, model, loss_fn, batch)
        # Update Model
        optim.zero_grad()
        log['loss'].backward()
        optim.step()
        for l in loss_name:
            logs[l].append(log[l].item())
    for l in loss_name:
        logs[l] = np.mean(logs[l])
    return logs

def test(args, model, loss_fn, loader):
    loss_name = ['loss', 'v_loss', 'a_loss', 'd_loss']
    logs = {l:[] for l in loss_name}
    model = model.to(args.device)
    model.eval()
    for batch in loader:
        log = step(args, model, loss_fn, batch)
        for l in loss_name:
            logs[l].append(log[l].item())
    for l in loss_name:
        logs[l] = np.mean(logs[l])
    return logs
