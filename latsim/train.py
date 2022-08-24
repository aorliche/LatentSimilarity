
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform, loguniform, randint

from latsim.util import *
from latsim.latsim import LatSim

def train(sim, nEpochs, Xt, yt, Xv=None, yv=None, 
        taskWeights=[1], 
        sparseLoss=[0], alignLoss=[0], disentangleLoss=[0],
        lr=1e-4, weight_decay=1e-4,
        validFnameTemplate=None,
        pPeriod=5, printProgress=True):
   
    mseLoss = torch.nn.MSELoss()
    ceLoss = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(sim.parameters(), lr=lr, weight_decay=weight_decay)

    nTasks = sim.w.shape[1]
    validLoss = [[] for _ in range(nTasks)]

    assert nTasks == len(taskWeights), "Not enough task weights for number of tasks"

    if not isinstance(sparseLoss, list):
        sparseLoss = nTasks*[sparseLoss]
    if not isinstance(alignLoss, list):
        alignLoss = nTasks*[alignLoss]
    if not isinstance(disentangleLoss, list):
        disentanbleloss = nTasks*[disentangleLoss]

    for epoch in range(nEpochs):
        optim.zero_grad()

        # Target loss
        res, zs = sim(Xt, yt, return_zs=True)
        loss = []
        avg = getAvg(res)
        for task in range(nTasks):
            if avg[task].dim() > 1:
                loss.append(taskWeights[task]*ceLoss(avg[task], yt[task]))
            else:
                loss.append(taskWeights[task]*mseLoss(avg[task], yt[task]))

        # Additional regularization
        if sparseLoss is not None and any(ls != 0 for ls in sparseLoss):
            loss += getSparseLoss(sim, sparseLoss)
        if alignLoss is not None and any(ls != 0 for ls in alignLoss):
            loss += getAlignLoss(sim, zs, alignLoss)
        if disentangleLoss is not None and any(ls != 0 for ls in disentangleLoss):
            loss += getDisentangleLoss(sim, disentangleLoss)

        # Step
        sum(loss).backward()
        optim.step()

        # Validation
        if epoch % pPeriod == 0 or epoch == nEpochs-1:
            if printProgress:
                print(f'epoch {epoch} loss={[float(ls) for ls in loss]}')
            if Xv is not None and yv is not None:
                Xtv = torch.cat([Xt, Xv])
                ytv = torch.cat([yt, yv])
                losses = validate(sim, Xtv, ytv, torch.arange(Xt.shape[0],Xt.shape[0]+Xv.shape[0]))
                for i,lss in enumerate(losses):
                    if (len(validLoss[i]) == 0 or 
                            (yy[i].dim() == 1 and lss < min(validLoss[i])) or 
                            (yy[i].dim() > 1 and lss > max(validLoss[i]))):
                        if printProgress:
                            print(f'New best validation epoch {epoch} {i} loss={lss}')
                        torch.save(sim.state_dict(), f'{validFnameTemplate}{i}.pyt')
                        validLoss[i].append(float(lss))
    if printProgress:
        print('Complete')

class LatSimEstimator(BaseEstimator):
    def __init__(self, **params):
        self.set_params(**params)

    @staticmethod
    def get_default_params():
        return dict(regOrClass='reg', standardize=True,
            dp=0.2, edp=0.1, wInit=1e-4, dim=2, temp=1,
            sparseLoss=0, alignLoss=0,
            nEpochs=200, lr=1e-4, weight_decay=1e-4)

    @staticmethod
    def get_default_distributions():
        return dict(
            standardize=[True, False],
            dp=[0,0.2,0.5], 
            edp=[0,0.2],
            #wInit=[1e-4,1e-3],
            dim=[1,2,10],
            #temp=[0.1,1,10],
            #sparseLoss=[0,1e-2,1e-1,1,10,100],
            #alignLoss=[0,1e-2,1e-1,1,10,100],
            lr=[1e-4,1e-3,1e-2,1e-1],
            weight_decay=[0,1e-4,1e-3]
        )

    def get_params(self, deep=False):
        return dict(regOrClass=self.regOrClass, standardize=self.standardize,
            dp=self.dp, edp=self.edp, wInit=self.wInit, dim=self.dim, temp=self.temp,
            sparseLoss=self.sparseLoss, alignLoss=self.alignLoss,
            nEpochs=self.nEpochs, lr=self.lr, weight_decay=self.weight_decay)

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        assert self.regOrClass == 'reg' or self.regOrClass == 'class', 'regOrClass must be reg or class'
        return self

    def fit(self, data, targets):
        assert data.ndim == 3, 'Data needs dimensions (nSamples,nMods,nFeat)'
        assert targets.ndim == 1, 'Targets need dimension=1 (nSamples)'
        self.Xt = torch.from_numpy(data).float().cuda()
        if self.standardize:
            self.mu = torch.mean(self.Xt, axis=0, keepdim=True)
            self.sd = torch.std(self.Xt, axis=0, keepdim=True)
            self.Xt = (self.Xt-self.mu)/self.sd
        if self.regOrClass == 'reg':
            self.yt = [torch.from_numpy(targets).float().cuda()]
        else:
            self.yt = [F.one_hot(torch.from_numpy(targets)).float().cuda()]
        self.sim = LatSim(1, self.Xt, dp=self.dp, edp=self.edp, 
            wInit=self.wInit, dim=self.dim, temp=self.temp)
        train(self.sim, self.nEpochs, self.Xt, self.yt,
            sparseLoss=self.sparseLoss, alignLoss=self.alignLoss, 
            lr=self.lr, weight_decay=self.weight_decay, 
            printProgress=False)
        return self

    def predict(self, data):
        n = self.Xt.shape[0]
        m = data.shape[0]
        testIdcs = torch.arange(n,n+m).long().cuda()
        Xtt = torch.cat([self.Xt, torch.from_numpy(data).float().cuda()])
        if self.standardize:
            Xtt = (Xtt-self.mu)/self.sd
        ytt = torch.cat([self.yt[0], torch.zeros(data.shape[0], self.yt[0].shape[1]).float().cuda()])
        ytt = [ytt]
        res = self.sim(Xtt, ytt, testIdcs)
        res = getAvg(res)[0][testIdcs]
        if res.ndim == 1:
            return res.detach().cpu().numpy()
        else:
            return np.argmax(res.detach().cpu().numpy(), axis=1)

def tune(Xt, yt, regOrClass, n_iter=100, cv=10, params=None, distribution=None):
    if params is None:
        params = LatSimEstimator.get_default_params()
    if distribution is None:
        distributions = LatSimEstimator.get_default_distributions()
    params['regOrClass'] = regOrClass
    sim = LatSimEstimator(**params)
    if regOrClass == 'reg':
        clf = RandomizedSearchCV(sim, distributions, cv=cv, n_iter=n_iter, scoring='neg_root_mean_squared_error')
    else:
        clf = RandomizedSearchCV(sim, distributions, cv=cv, n_iter=n_iter, scoring='accuracy')
    search = clf.fit(Xt, yt)
    return search
