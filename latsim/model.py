
'''
Main LatentSimilarity class
Modified 1/12/23 for simplicity

Best way to use is to define a stopping criteria
Use of the sklearn interface results in automatic creation of validation set,
from the training set, which then determines best parameter choice

2/8/23 Added automatic validation set creation in sklearn.py
2/9/23 Automatic fallback to CPU if CUDA not available
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def to_cuda(x):
    try:
        return x.float().cuda()
    except:
        return x.float()

class LatSim(nn.Module):
    def __init__(self, d, ld=2):
        super(LatSim, self).__init__()
        self.A = nn.Parameter(to_cuda(torch.randn(d,ld)/(d**0.5)))

    def E(self, xtr, xt):
        # Normalize inputs
        mu = torch.mean(xtr, dim=0, keepdims=True)
        xtr = xtr - mu
        xt = xt - mu
        # Similarity matrix
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
    kwargs['lossfn'] = nn.MSELoss()
    train_sim(*args, **kwargs)

def train_sim_ce(*args, **kwargs):
    kwargs['lossfn'] = nn.CrossEntropyLoss()
    train_sim(*args, **kwargs)

def train_sim(sim, xtr, ytr, stop=0, xv=None, yv=None, lr=1e-4, wd=1e-4, nepochs=1000, pperiod=20, lossfn=nn.MSELoss(), verbose=False):
    # Validation set   
    if xv is not None and yv is not None:
        if yv.dim() == 2:
            yvv = torch.argmax(yv, dim=1)
            best = 0
        else:
            best = float('inf')
        bestA = None

    # Optimizers
    optim = torch.optim.Adam(sim.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=20, factor=0.75, eps=1e-7) 

    for epoch in range(nepochs):
        optim.zero_grad()
        yhat = sim(xtr, ytr)
        loss = lossfn(yhat, ytr)
        loss.backward()
        optim.step()
        if loss < stop:
            break
        sched.step(loss)
        # Check validation set
        if xv is not None and yv is not None:
            with torch.no_grad():
                yhat = sim(xtr, ytr, xv)
                if yv.dim() == 2:
                    yhat = torch.argmax(yhat, dim=1)
                    acc = torch.sum(yvv == yhat)/len(yvv)
                    acc = float(acc)
                    better = acc > best
                else:
                    acc = lossfn(yhat, yv)
                    acc = float(acc)
                    better = acc < best
                if better:
                    best = acc
                    bestA = sim.A.detach()
                    if verbose:
                        print(f'Best acc {acc}') 
        if epoch % pperiod == 0 or epoch == nepochs-1:
            if verbose:
                print(f'{epoch} loss: {float(loss)} lr: {sched._last_lr}')
        optim.zero_grad()

    if xv is not None and yv is not None:
        if verbose:
            print(f'Final best acc {best}')
        sim.A = nn.Parameter(to_cuda(bestA))
    if verbose:
        print('Complete')
        
