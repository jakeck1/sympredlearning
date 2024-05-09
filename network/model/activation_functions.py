import numpy as np

def sigmoid(x):
    "Numerically-stable sigmoid function."
    where = (x>0)

    out = np.zeros_like(x)
    out[where] = 1.0/(1.0+np.exp(-x[where]))
    out[~where] = np.exp(x[~where])/(1.0 + np.exp(x[~where]))

    return out


class ActivationFunction:


    def __init__(self):
        pass

    def apply(self,input):

        raise NotImplementedError

    def apply_derivative(self,input):
        raise NotImplementedError


class IdentityActivation(ActivationFunction):

    def __init__(self):
        super().__init__()

    def apply(self,input):
        return input

    def apply_derivative(self,input):
        return np.ones_like(input)


class ReLu(ActivationFunction):

    def __init__(self):
        super().__init__()

    def apply(self,input):

        return input*(input>0)

    def apply_derivative(self,input):
        return np.ones_like(input)*(input>0.0).astype(float)


class Sigmoid(ActivationFunction):

    def __init__(self):
        super().__init__()

    def apply(self, input):
        return sigmoid(input)

    def apply_derivative(self, input):

        s = sigmoid(input)
        return s*(1.0-s)


class Tanh(ActivationFunction):

    def __init__(self):
        super().__init__()

    def apply(self,input):

        return np.tanh(input)

    def apply_derivative(self,input):

        return 1.0 - np.tanh(input)**2


class LeakyRELU(ActivationFunction):

    def __init__(self,alpha=1e-3):
        self.alpha = alpha
        super().__init__()

    def apply(self,input):

        return input*((input>0).astype(float)+self.alpha*(input<0).astype(float))


    def apply_derivative(self,input):

        return np.ones_like(input)*((input>0).astype(float)+self.alpha*(input<0).astype(float))



