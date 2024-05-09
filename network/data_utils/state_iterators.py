import numpy as np


class StateSpaceIterator:



    def __iter__(self):
        self.max_iter = None
        pass

    def __next__(self):
        raise StopIteration



class DiscreteRandomWalkIterator(StateSpaceIterator):


    def __init__(self,P,initial_state = 'random',max_iter = None):
        super().__init__()
        self.states = np.arange(P.shape[0])
        self.P = P
        if initial_state=='random':
            initial_state = np.random.choice(self.states)

        self.state = initial_state
        self.max_iter = max_iter
    def reset_random(self):
        self.state = np.random.choice(self.states)


    def __next__(self):

        p = self.P[self.state]

        next_state = np.random.choice(self.states,p=p)

        self.state = next_state

        return next_state



class DiscreteTimeGaussianProcessIterator(StateSpaceIterator):


    def __init__(self,A,Sigma = None,initial_state = 'zero',max_iter = None):
        super().__init__()


        self.A = A
        self.state_dim = A.shape[0]


        if isinstance(Sigma,float):
            Sigma = np.eye(self.state_dim)*Sigma
        elif Sigma is None:
            Sigma = np.eye(self.state_dim)

        self.Sigma = Sigma

        if initial_state =='zero':
            initial_state = np.zeros(self.state_dim)
        elif initial_state == 'random':
            initial_state = np.random.multivariate_normal(np.zeros(self.state_dim),self.Sigma)
        self.state = initial_state
        self.max_iter = max_iter

    def __next__(self):

        next_state = self.A@self.state + np.random.multivariate_normal(np.zeros(self.state_dim),self.Sigma)

        self.state = next_state

        return next_state



