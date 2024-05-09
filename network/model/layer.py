import numpy as np
from .weights import EmptyWeights
from .activation_functions import IdentityActivation
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

class SingleSRLayer:



    def __init__(self,
                 n_cells = None,
                 recurrent_weights = None,
                 forward_weights = None,
                 backward_weights=None,
                 tau = 0.0,
                 gamma_r=0.0,
                 gamma_f=0.0,
                 gamma_b = 0.0,
                 activation_function = None,
                 normalize_gammas=True,
                 forward_euler=False):

        if recurrent_weights is None and forward_weights is None and backward_weights is None and n_cells is None:
            raise ValueError('At least one of recurrent_weights,forward weights,backward weights has to be provided if n_cells is not specified')

        n_ins = np.array([self._weight_helper(recurrent_weights),self._weight_helper(forward_weights),self._weight_helper(backward_weights),n_cells])
        #print(n_ins)
        n_ins = n_ins[n_ins !=None]
        assert len(np.unique(n_ins))== 1, 'The dimensions of the provided weights and/or n_cells do not seem to match'

        self.n_cells = n_ins[0]

        if forward_weights  is None:
            forward_weights = EmptyWeights(self.n_cells,0)
        if backward_weights is None:
            backward_weights = EmptyWeights(self.n_cells, 0)
        if recurrent_weights is None:
            recurrent_weights = EmptyWeights(self.n_cells,0)

        self.forward_weights = forward_weights
        self.backward_weights = backward_weights
        self.recurrent_weights = recurrent_weights

        self.gamma_r = gamma_r
        self.gamma_f = gamma_f
        self.gamma_b = gamma_b
        self.gamma_l = 1.0

        if normalize_gammas:
            # we do this normalizationa bit awkwardly to achieve the sum of gammas =1,
            # but we also want to keep the default behaviour for a single gamma not equal 1
            n_nonzero = np.sum([gamma>0 for gamma in [gamma_r,gamma_f,gamma_b]])

            self.gamma_r/=n_nonzero
            self.gamma_f /= n_nonzero
            self.gamma_b /= n_nonzero
            self.gamma_l = 1.0 - (self.gamma_r+self.gamma_f+self.gamma_b)

        self.activity = np.zeros(self.n_cells)
        self.tau = tau

        if activation_function is None:
            activation_function = IdentityActivation()

        self.activation_function = activation_function

        self.forward_euler = forward_euler




    def update_forward_euler(self,lateral_input, bottom_up_input,top_down_input,dt):
        total_input = self.gamma_r * self._output_helper(self.W_r,
                                                         self.activity) + self.gamma_f * bottom_up_input + self.gamma_b * top_down_input + self.gamma_l * lateral_input
        delta_activity = -self.activity + self.activation_function.apply(total_input)

        new_activity = self.activity + delta_activity * dt/self.tau

        self.activity = new_activity


    def update_by_equilibrium(self, lateral_input, bottom_up_input, top_down_input):


        non_recurrent_input =  self.gamma_f * bottom_up_input + self.gamma_b * top_down_input + self.gamma_l*lateral_input

        equilibrium_activity = np.linalg.solve(np.eye(self.n_cells)-self.gamma_r*self.W_r,non_recurrent_input)

        self.activity=equilibrium_activity

    def update_backward_euler(self,lateral_input,bottom_up_input,top_down_input,dt):
        non_recurrent_input = self.gamma_f * bottom_up_input + self.gamma_b * top_down_input + self.gamma_l * lateral_input

        if isinstance(self.activation_function, IdentityActivation) :

            input_plus_activity = self.tau*self.activity + dt*non_recurrent_input

            I = np.eye(self.n_cells)

            M = self.tau*I + dt*(I-self.gamma_r*self.W_r)

            new_activity = np.linalg.solve(M,input_plus_activity)

        else:

            current_activity =self.activity.copy()
            def right_hand_side(next_activity):
                return next_activity - (current_activity + (dt/self.tau) * (-next_activity + self.activation_function.apply(np.dot(self.gamma_r*self.W_r,next_activity) + non_recurrent_input)))

            def right_hand_side_jacobian(next_activity):
                return np.eye(next_activity.shape[0])- (dt/self.tau)*(-np.eye(next_activity.shape[0])+self.activation_function.apply_derivative(np.dot(self.gamma_r*self.W_r,next_activity) + non_recurrent_input)*self.gamma_r*self.W_r)
            new_activity = fsolve(right_hand_side, current_activity,fprime=right_hand_side_jacobian)

        self.activity = new_activity


    def update(self,dt,lateral_input = None, bottom_up_input = None,top_down_input = None):
        lateral_input, bottom_up_input, top_down_input = self._input_helper(lateral_input), self._input_helper(bottom_up_input), self._input_helper(top_down_input)

        if self.tau >0:
            if self.forward_euler:
                self.update_forward_euler(lateral_input,bottom_up_input,top_down_input,dt)
            else:

                self.update_backward_euler(lateral_input,bottom_up_input,top_down_input,dt)


        elif self.tau == 0.0 and isinstance(self.activation_function,IdentityActivation):
            self.update_by_equilibrium(lateral_input,bottom_up_input,top_down_input)

        else:
            raise NotImplementedError
        return self.activity


    def forward(self):
        return self._output_helper(self.W_f, self.activity)

    def backward(self):
        return self._output_helper(self.W_b, self.activity)


    def _input_helper(self,x):
        return x if (x is not None) else np.zeros(self.n_cells)

    def _output_helper(self,W,x):

        if isinstance(W,float):
            if W == 0.0:
                return 0.0
            else:
                return W*x
        else:

            return W@x

    def _weight_helper(self,weight):
        try:
            return weight.n_in
        except:
            return None

    @property
    def W_f(self):
        return self.forward_weights.W

    @property
    def W_b(self):
        return self.backward_weights.W

    @property
    def W_r(self):
        return self.recurrent_weights.W
