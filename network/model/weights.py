import numpy as  np
from .learning_rules import LearningRule,EmptyLearningRule


class EmptyWeights:


    def __init__(self,n_in,n_out):

        self.W = 0.0
        self.n_in = n_in
        self.n_out = n_out
        self.learning_rule = EmptyLearningRule()
        self.tau = 1.0

    def apply_learning_rule(self,post_new,post_old,pre_new,pre_old):

        return self.learning_rule.weight_update(self.W, post_new, post_old, pre_new, pre_old)

    def update(self,dt,post_new,post_old,pre_new,pre_old):
        self.W += self.apply_learning_rule(post_new,post_old,pre_new,pre_old)*dt/self.tau


class SynapticWeights(EmptyWeights):


    def __init__(self,n_in,n_out,tau=0.0,learning_rule = None,initial=None):

        super().__init__(n_in,n_out)
        if isinstance(initial,np.ndarray):
            self.W = initial.copy()
        elif isinstance(initial,WeightInitializer):

            self.W = initial.initialize_weights((n_out,n_in))

        elif initial is None:
            transform = PowerTransform(2)
            initializer = NormalInitializer(transform = transform)
            self.W = initializer.initialize_weights((n_out,n_in))
        else:
            raise ValueError

        if learning_rule is not None:
            assert isinstance(learning_rule,LearningRule)
            self.learning_rule = learning_rule

        self.tau = tau





class WeightInitializer:


    def __init__(self):
        pass

    def initialize_weights(self,shape):

        raise NotImplementedError


class ZeroWeightInitializer(WeightInitializer):


    def initialize_weights(self,shape):
        return np.zeros(shape)

class NormalInitializer(WeightInitializer):


    def __init__(self,scaler = None, transform = None):

        if scaler is None:
            scaler = PowerScaler(-3)

        self.scaler = scaler
        if transform is None:
            transform = IdentityTransform()

        self.transform = transform


    def initialize_weights(self,shape):

        scale = self.scaler.get_scale(shape)


        return self.transform.apply(np.random.normal(size= shape,scale = scale))

class UniformInitializer(WeightInitializer):


    def __init__(self,low,high,transform = None):
        self.low = low
        self.high = high
        if transform is None:
            transform = IdentityTransform()

        self.transform = transform

    def initialize_weights(self,shape):

        return self.transform.apply(np.random.uniform(self.low,self.high,size=shape))



class Scaler:


    def __init__(self):
        pass

    def get_scale(self,shape):
        raise NotImplementedError


class PowerScaler(Scaler):

    def __init__(self,n):
        self.n = n

    def get_scale(self,shape):

        return max(shape)**self.n






class Transform:


    def __init__(self):
        pass

    def apply(self,x):
        raise NotImplementedError


class IdentityTransform(Transform):
    def __init__(self):
        pass

    def apply(self,x):
        return x


class AbsTransform(Transform):
    def __init__(self):
        pass
    def apply(self,x):
        return np.abs(x)

class PowerTransform(Transform):
    def __init__(self,n):
        self.n = n

    def apply(self,x):
        return x **self.n
