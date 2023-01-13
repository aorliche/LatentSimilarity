
'''
Main LatentSimilarity class
Modified 1/12/23 for simplicity

Fast, but requires you to know a target stopping criteria
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LatSim(nn.Module):
    def __init__(self, d, ld=2):
        super(LatSim, self).__init__()
        self.A = nn.Parameter((torch.rand(ld,d)/(d**0.5)).float().cuda())

    def E(self, xtr, xt):
        AT = (xtr@self.A).T
        A = xt@self.A
        E = A@AT
        return F.softmax(E,dim=1)
        
    def forward(self, xtr, ytr, xt=None):
        if xt is None:
            xt = xtr
        E = self.E(xtr, xt)
        return E@ytr

def train_sim_mse(*args, **kwargs):
    train_sim(*args, **kwargs)

def train_sim_ce(*args, **kwargs):
    kwargs['lossfn'] = nn.CrossEntropyLoss()
    train_sim(*args, **kwargs)

def train_sim(sim, xtr, ytr, stop, lr=1e-4, nepochs=100, pperiod=20, lossfn=nn.mseLoss(), verbose=False):
    # Optimizers
    optim = torch.optim.Adam(latsim.parameters(), lr=lr, weight_decay=0)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=20, factor=0.75, eps=1e-7)

    for epoch in range(nepochs):
        optim.zero_grad()
        yhat = latsim(xtr, xtr, ytr)
        loss = lossfn(yhat, ytr)
        loss.backward()
        optim.step()
        if loss < stop:
            break
        sched.step(loss)
        if verbose:
            if epoch % pperiod == 0 or epoch == nepochs-1:
                print(f'{epoch} recon: {float(loss)} lr: {float(sched._last_lr)}')

    optim.zero_grad()
    if verbose:
        print('Complete')
