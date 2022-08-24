
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations, product

def flatten(lst):
    [a for b in lst for a in b]

def mask(e):
    return e - torch.diag(torch.diag(e.detach()))

def arith(n):
    return int(n*(n+1)/2)

def allBelowThresh(losses, thresh):
    for loss,thr in zip(losses, thresh):
        if loss > thr:
            return False
    return True

def getDisentangleLoss(sim, disParam):
    nMods = sim.w.shape[0]
    nTasks = sim.w.shape[1]
    dims = sim.w.shape[3]
    tasks = list(combinations(np.arange(nTasks), 2))
    loss = []
    for ta,tb in tasks:
        for mod in range(nMods):
            for dim in range(dims):
                loss.append(disParam*torch.sum(torch.abs(sim.w[mod,ta,:,dim]*sim.w[mod,tb,:,dim])))
    return loss

def getSparseLoss(sim, sparseParamPerTask):
    ''' 
    # L1
    nTasks = sim.w.shape[1]
    loss = []
    for task in range(nTasks):
        loss.append(sparseParamPerTask[task]*torch.sum(torch.abs(sim.w[:,task,:,:])))
    return loss
    '''
    # Sparsity through entropy reg
    nMods = sim.w.shape[0]
    nTasks = sim.w.shape[1] 
    loss = []
    for mod in range(nMods):
        for task in range(nTasks):
            mag = torch.abs(sim.w[mod,task])
            maxMag = torch.max(mag, keepdim=True, dim=1).values
            p = mag/maxMag
            loss.append(-sparseParamPerTask[task]*torch.sum(p*torch.log(p+1e-10)))
    return loss

def getAlignLoss(sim, zs, alignParamPerTask):
    nMods = sim.w.shape[0]
    mods = list(combinations(np.arange(nMods), 2))
    loss = []
    for task in range(3):
        for moda,modb in mods:
            loss.append(alignParamPerTask[task]*torch.sum((zs[moda,task,:,:]-zs[modb,task,:,:])**2))
    return loss

def getAvg(res):
    nMods = len(res)
    nTasks = len(res[0])
    avg = nTasks*[0]
    for mod in range(nMods):
        for task in range(nTasks):
            avg[task] += res[mod][task]/nMods
    return avg

def validate(sim, X, ys, testIdcs):
    mseLoss = torch.nn.MSELoss()
    sim.eval()
    losses = []
    with torch.no_grad():
        res = sim(X, ys, testIdcs)
        for r,y in zip(getAvg(res), ys):
            if y.dim() == 1:
                loss = mseLoss(r[testIdcs], y[testIdcs]).cpu().numpy()**0.5
                losses.append(loss)
            else:
                corr = (torch.argmax(r, dim=1) == torch.argmax(y, dim=1))[testIdcs]
                loss = torch.sum(corr)/len(testIdcs)
                losses.append(loss)
    sim.train()
    return losses
