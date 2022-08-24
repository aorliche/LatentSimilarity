
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from latsim.util import *

class LatSim(nn.Module):
    def __init__(self, nTasks, inp, dp=0.5, edp=0.1, wInit=1e-4, dim=2, temp=1):
        super(LatSim, self).__init__()
        self.nMods = inp.shape[1]
        self.nTasks = nTasks
        self.w = nn.Parameter(wInit*torch.randn(inp.shape[1],nTasks,inp.shape[-1],dim).float().cuda())
        self.dp = nn.Dropout(p=dp)
        self.edp = nn.Dropout(p=edp)
        self.temp = temp if isinstance(temp, list) else nTasks*[temp]
    
    def getLatentsAndEdges(self, x, mod, task, w=None):
        if w is None:
            w = self.w[mod,task]
        e = 1e-10
        z = x@w
        e = e+z@z.T
        return z, e
        
    def forward(self, x, ys, testIdcs=None, return_es=False, return_zs=False):
        assert len(ys) == self.nTasks, 'ys list does not have nTasks tasks'
        x = self.dp(x)
        res = []
        es = []
        zs = []
        for mod in range(self.nMods):
            res.append([])
            es.append([])
            zs.append([])
            for task,y in enumerate(ys):
                z, e = self.getLatentsAndEdges(x[:,mod], mod, task)
                zs[-1].append(z)
                if testIdcs is not None:
                    e[:,testIdcs] = 0
                e = mask(e)
                e = self.edp(e)
                e[e == 0] = float('-inf')
                e = F.softmax(e/self.temp[task], dim=1)
                res[-1].append(e@y)
                es[-1].append(e.clone())
        if return_es and return_zs:
            return res, es, zs
        elif return_es:
            return res, es
        elif return_zs:
            return res, zs
        else:
            return res
