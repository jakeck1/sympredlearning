import numpy as np
import numpy.random as npr
import neuronav.utils as utils
from neuronav.agents.td_agents import TDSR

def normalize_with_threshold(arr, threshold=1e-9):
    """
    Normalize a numpy array to unit norm, except if its norm is close to zero.

    Parameters:
    - arr: numpy array to be normalized.
    - threshold: a small value below which the norm is considered close to zero.

    Returns:
    - The normalized array if its norm is above the threshold, otherwise the original array.
    """
    norm = np.linalg.norm(arr)
    if norm < threshold:
        # The norm is considered close to zero, so return the original array.
        return arr
    else:
        # Normalize the array to unit norm.
        return arr / norm


class TDSR_WM(TDSR):

    '''td-SR agent using a deterministic world model'''

    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 1e-1,
        gamma: float = 0.99,
        poltype: str = "softmax",
        beta: float = 1e4,
        epsilon: float = 1e-1,
        M_init=None,
        weights: str = "direct",
        goal_biased_sr: bool = True,
        bootstrap: str = "max-min",
        w_value: float = 1.0,
        forward_weight: float = 1.0,
        backward_weight: float = 0.0,
        learn_SR: bool = True,
        learn_reward: bool =True,
        learn_model: bool = True,
        T: np.array = None):


        super().__init__(state_size=state_size,
        action_size=action_size,
        lr=lr,
        gamma=gamma,
        poltype=poltype,
        beta=beta,
        epsilon=epsilon,
        weights=weights,
        goal_biased_sr=goal_biased_sr,
        bootstrap=bootstrap,
        w_value=w_value,)

        self.forward_weight = forward_weight
        self.backward_weight = backward_weight
        #assert forward_weight+backward_weight !=0
        self.normalizing_factor = self.forward_weight + self.backward_weight

        self.learn_SR = learn_SR
        self.learn_reward = learn_reward
        self.learn_model = learn_model
        self.learn_T = False
        if T is None:
            T = np.zeros([action_size, state_size, state_size],dtype = float)
            self.learn_T = True
        self.T = T
        if M_init is None:
            self.M = np.eye(state_size)
        elif np.isscalar(M_init):
            self.M = np.random.randn(state_size,state_size)*M_init
        else:
            self.M = M_init
        self.visited = np.zeros((action_size, state_size), dtype=bool)  # Keep track of visited state-action pairs


    def q_estimate(self, state):

        return self.K[:, state, :] @ self.w




    def _update(self, current_exp, **kwargs):
        s, s_a, s_1, r, d = current_exp
        return_update = kwargs.get('return_update',False)
        if self.learn_SR:
                 m_error = self.update_sr(s, s_a, s_1, d, **kwargs)
        if self.learn_reward:
                w_error = self.update_w(s, s_1, r)
        if self.learn_model:
                t_error = self.update_t(s,s_a,s_1)
        q_error = self.q_error(s, s_a, s_1, r, d)

        if return_update:

            return m_error
        else:
            return q_error


    def update_t(self, s, s_a, s_1):

        if not self.learn_T:
            return None
        if not self.visited[s_a, s]:

            #next_onehot = utils.onehot(s_1, self.state_size)
            self.T[s_a, s,s_1] = 1.0 #next_onehot
            self.visited[s_a, s] = True
        return None




    def update_sr(self, s, s_a, s_1, d, next_exp=None, prospective=False,normalize_update = False, return_update = True):


            alpha = self.forward_weight
            beta = self.backward_weight

            if d:
                m_error_forward = (
                        utils.onehot(s, self.state_size) + (self.gamma/(1-self.gamma)) * utils.onehot(s_1, self.state_size) - self.M[s, :])
            else:
                m_error_forward = utils.onehot(s, self.state_size) + self.gamma * self.M[s_1,:] - self.M[s, :]
            m_error_backward = utils.onehot(s_1, self.state_size) + self.gamma * self.M[s, :] - self.M[s_1, :]

            if not prospective:
                # actually perform update to SR if not prospective
                if normalize_update:
                    total_update= np.vstack((m_error_forward,m_error_backward))
                    #we calculate norm of the stacked updates, and account for the special case where the states are the same
                    total_update_norm = np.linalg.norm(total_update)

                    if s == s_1:
                        # norm(x,x) = sqrt(2) norm(x), but norm(2x) = 2 norm(x)
                        total_update_norm*= np.sqrt(2)

                    if total_update_norm>=1e-12:
                        m_error_forward = m_error_forward/total_update_norm
                        m_error_backward = m_error_backward/total_update_norm

                    m_error_forward = m_error_forward*normalize_update
                    m_error_backward = m_error_backward*normalize_update




                self.M[s, :] += alpha*self.lr * m_error_forward
                self.M[s_1,:] += beta*self.lr * m_error_backward


            return alpha*m_error_forward+beta*m_error_backward



    def get_policy(self, M=None, goal=None):
        if goal is None:
            goal = self.w

        if M is None:
            M = self.M

        Q = np.einsum('ijk,kl',self.T,M).dot(goal)

        return self.base_get_policy(Q)

    def get_M_states(self):

        return self.M
    @property
    def K(self):
        return np.einsum('ijk,kl',self.T,self.M)
    @property
    def Q(self):
        return np.einsum('...i,...i',self.K,self.w)





