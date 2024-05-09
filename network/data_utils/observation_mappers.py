import numpy as np


class ObservationMapper:

    def __init__(self):
        pass

    def sample(self,state):

        raise NotImplementedError


class DiscreteObservationMapper(ObservationMapper):


    def __init__(self,Phi,noise =0.0):

        self.Phi = Phi
        self.noise = noise
        self.in_dim = self.Phi.shape[1]
        self.out_dim = self.Phi.shape[0]

    def sample(self,state):
        epsilon = 0.0
        if self.noise !=0:
            epsilon = np.random.multivariate_normal(np.zeros(self.out_dim),self.noise*np.eye(self.out_dim))
        return self.Phi[:,state]+epsilon

class GaussianObservationMapper(ObservationMapper):
    def __init__(self, Phi, Sigma=0.0):
        self.Phi = Phi


        self.in_dim = self.Phi.shape[1]
        self.out_dim = self.Phi.shape[0]

        if isinstance(Sigma,float):
            Sigma = Sigma*np.eyes(self.out_dim)
        self.Sigma = Sigma

    def sample(self, state):
        return self.Phi@state + np.random.multivariate_normal(np.zeros(self.out_dim),self.Sigma)

