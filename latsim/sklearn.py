'''
sklearn interface for e.g., parameter search
In the new version, you call GridSearchCV or others yourself
'''

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.base import BaseEstimator

from latsim import LatSim, train_sim_mse, train_sim_ce

def to_torch(x):
    if not isinstance(x, torch.Tensor):
        return torch.from_numpy(x).float().cuda()
    else:
        return x

'''
One class for regression, one (sub)class for classification
'''
class LatSimReg(BaseEstimator):
    def __init__(self, **params):
        self.set_params(**params)

    @staticmethod
    def get_default_params():
        return dict(ld=2, stop=0, lr=1e-4, nepochs=100)

    @staticmethod
    def get_default_distributions():
        return dict(
            ld=[1,2,10],
            stop=[0,1,10,100],
            lr=[1e-5,1e-4,1e-2],
            nepochs=[100,1000,10000],
        )

    def get_params(self, deep=False):
        return dict(ld=self.ld, stop=self.stop, lr=self.lr, nepochs=self.nepochs)

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, x, y, **kwargs):
        x = to_torch(x)
        y = to_torch(y)
        self.x = 1*x
        self.y = 1*y
        params = LatSimReg.get_default_params()
        for arg in kwargs:
            if arg in params:
                params[arg] = kwargs[arg]
        self.sim = LatSim(x.shape[1], params['ld'])
        del params['ld']
        train_sim_mse(self.sim, self.x, self.y, **params)
        return self

    def predict(self, x):
        x = torch.from_numpy(x).float().cuda()
        with torch.no_grad():
            yhat = self.sim(self.x, self.y, x)
        return yhat.detach().cpu().numpy()

'''
Classification subclass
'''
class LatSimClf(LatSimReg):
    def fit(self, x, y, **kwargs):
        y = to_torch(y).long()
        y = F.one_hot(y).float()
        return super().fit(x, y, **kwargs)

    def predict(self, x):
        yhat = super().predict(x)
        return np.argmax(yhat, axis=1)

# RandomizedSearchCV(sim, distributions, cv=cv, n_iter=n_iter, scoring='neg_root_mean_squared_error')
# RandomizedSearchCV(sim, distributions, cv=cv, n_iter=n_iter, scoring='accuracy')
