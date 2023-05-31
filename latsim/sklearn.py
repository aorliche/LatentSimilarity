'''
sklearn interface for e.g., parameter search
In the new version, you call GridSearchCV or others yourself

Validation set created by default from training data
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from latsim.model import LatSim, train_sim, to_cuda

def to_torch(x):
    if not isinstance(x, torch.Tensor):
        return to_cuda(torch.from_numpy(x))
    else:
        return x

def np_one_hot(y):
    y = y.astype('int')
    r = np.zeros((y.size, y.max()+1))
    r[np.arange(y.size), y] = 1
    return r

'''
One class for regression, one (sub)class for classification
'''
class LatSimReg(BaseEstimator):
    def __init__(self, **params):
        self.set_params(**params)

    @staticmethod
    def get_default_params():
        return dict(ld=2, stop=0, lr=1e-4, wd=1e-4, nepochs=100, lossfn=nn.MSELoss(), verbose=False, clf=False)

    @staticmethod
    def get_default_distributions():
        return dict(
            ld=[1,2,10],
            stop=[0,1,10*10,100*100],
            lr=[1e-5,1e-4,1e-3],
            wd=[1e-5,1e-4,1e-3],
            nepochs=[100,1000,2000],
        )

    def get_params(self, **params):
        return dict(ld=self.ld, stop=self.stop, lr=self.lr, wd=self.wd, nepochs=self.nepochs, lossfn=self.lossfn, verbose=self.verbose, clf=self.clf)

    def set_params(self, **params):
        dft = LatSimReg.get_default_params()
        for key in dft:
            if key in params:
                setattr(self, key, params[key])
            else:
                setattr(self, key, dft[key])
        return self

    def fit(self, x, y, **kwargs):
        # Load parameters
        params = self.get_params()
        for arg in kwargs:
            if arg in params:
                params[arg] = kwargs[arg]
        self.x, self.y = x, y
        # Make automatic validation sets
        if params['clf']:
            self.x, self.xv, self.y, self.yv = train_test_split(x, y, stratify=y, train_size=0.75)
            self.y = np_one_hot(self.y)
            self.yv = np_one_hot(self.yv)
        else:
            self.x, self.xv, self.y, self.yv = train_test_split(x, y, train_size=0.75)
        # Convert to torch
        self.x = to_torch(self.x)
        self.y = to_torch(self.y)
        self.xv = to_torch(self.xv)
        self.yv = to_torch(self.yv)
        # Create model
        self.sim = LatSim(x.shape[1], params['ld'])
        if params['verbose']:
            print(params)
        del params['ld']
        del params['clf']
        #train_sim(self.sim, self.x, self.y, **params)
        train_sim(self.sim, self.x, self.y, xv=self.xv, yv=self.yv, **params)
        return self

    def predict(self, x):
        x = to_cuda(torch.from_numpy(x))
        with torch.no_grad():
            yhat = self.sim(self.x, self.y, x)
        return yhat.detach().cpu().numpy()

class LatSimClf(LatSimReg):
    def LatSimClf(self, **params):
        self.set_params(**params)

    @staticmethod
    def get_default_params():
        return dict(ld=2, stop=0, lr=1e-4, wd=1e-4, nepochs=100, lossfn=nn.CrossEntropyLoss(), verbose=False, clf=True)

    @staticmethod
    def get_default_distributions():
        return dict(
            ld=[1,2,10],
            stop=[0,0.01,0.1,0.2,0.3],
            lr=[1e-5,1e-4,1e-3],
            wd=[1e-5,1e-4,1e-3],
            nepochs=[100,1000,2000],
        )

    def set_params(self, **params):
        dft = LatSimClf.get_default_params()
        for key in dft:
            if key in params:
                setattr(self, key, params[key])
            else:
                setattr(self, key, dft[key])
        return self

    def fit(self, x, y, **kwargs):
        return super().fit(x, y, **kwargs)

    def predict(self, x):
        yhat = self.predict_proba(x)
        return np.argmax(yhat, axis=1)
    
    def predict_proba(self, x):
        yhat = super().predict(x)
        return yhat
