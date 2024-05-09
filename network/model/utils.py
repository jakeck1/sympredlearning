import numpy as np
from .model import MultilayerSRNetwork
from .layer import SingleSRLayer
from .weights import SynapticWeights
from .learning_rules import  NormalizingLearningRule
import warnings

def create_model(n_cells,
                 learning_params,
                 tau_learning=0.08,
                 tau_activity=0.0,
                 gamma_r=0.7,
                 gamma_f = None,
                 tau_learning_f = None,
                 weight_initializer_W = None,
                 weight_initializer_V = None,
                 activation_functions = (None,None),
                 normalize_gammas = True,
                 forward_euler = False):
    n_cells_layer_1 = n_cells[0]
    n_cells_layer_2 = n_cells[1]

    ac_f_1 = activation_functions[0]
    ac_f_2 = activation_functions[1]

    if tau_learning_f is None:
        tau_learning_f = tau_learning
    if gamma_f is None:
        gamma_f = gamma_r

    alpha_1,beta_1 = learning_params[0]
    alpha_2,beta_2 = learning_params[1]
    W_learning_rule = NormalizingLearningRule(alpha=alpha_1, beta=beta_1)
    V_learning_rule = NormalizingLearningRule(alpha=alpha_2, beta=beta_2)

    W = SynapticWeights(tau=tau_learning, learning_rule=W_learning_rule, n_in=n_cells_layer_1, n_out=n_cells_layer_1,initial=weight_initializer_W)
    V = SynapticWeights(tau=tau_learning_f, learning_rule=V_learning_rule, n_in=n_cells_layer_1, n_out=n_cells_layer_2,initial=weight_initializer_V)

    rec_layer = SingleSRLayer(tau=tau_activity, gamma_r=gamma_r, recurrent_weights=W, forward_weights=V,activation_function=ac_f_1,normalize_gammas=normalize_gammas,forward_euler=forward_euler)
    out_layer = SingleSRLayer(tau=tau_activity, gamma_f=gamma_f, n_cells=n_cells_layer_2,activation_function=ac_f_2,normalize_gammas=normalize_gammas,forward_euler=forward_euler)

    model = MultilayerSRNetwork([rec_layer, out_layer])

    return model



def convergence_diffs(W,V,alpha_1,beta_1,alpha_2,beta_2,P,Phi_1,Phi_2,gamma_r,gamma_f):



    Pi = np.diag(get_stationary_dist(P))
    c_1 = alpha_1 + beta_1
    c_2 = alpha_2 + beta_2

    J_1 = ((alpha_1/c_1)* P.T@Pi + (beta_1/c_1)*Pi@P)

    J_2 = ((alpha_2 / c_2) * P.T @ Pi + (beta_2 / c_2) * Pi @ P)

    diff_W = W@Phi_1@Pi@Phi_1.T - Phi_1 @J_1 @Phi_1.T

    Q = Phi_1 @(Pi - gamma_f* J_2) @ Phi_1.T
    R = np.eye(W.shape[0])-gamma_r*W

    #inversion but cheaper
    try:
        S = np.linalg.solve(R,Q)
    except:
        S, _, _, _ = np.linalg.lstsq(R, Q, rcond=None)
        warnings.warn('Had to use leastsquares instead of exact solution. This might indicate that the recurrent matrix W behaves unexpectedly',RuntimeWarning)

    T = Phi_2@J_2@Phi_1.T
    diff_V = V@S - T
    return diff_W, diff_V





def get_convergence_targets(alpha_1,beta_1,alpha_2,beta_2,P,Phi_1,Phi_2,gamma_r,gamma_f):

    Pi = np.diag(get_stationary_dist(P))

    Pi_inv = np.linalg.inv(Pi)
#    print(Pi,'Pi')
  #  print(Pi_inv,'Pi_inv')



    P_reverse = Pi_inv @ P.T @ Pi
 #   print('P_rev',P_reverse)
    P_1 = P * alpha_1 /(alpha_1 + beta_1) + P_reverse * beta_1/(alpha_1 + beta_1)


    P_2 = P * alpha_2 /(alpha_2 + beta_2) + P_reverse * beta_2/(alpha_2 + beta_2)

    target_W = Phi_1 @ P_1.T  # @ np.linalg.inv(A)

    SR_1 = np.linalg.inv(np.eye(Phi_1.shape[1]) - gamma_r * P_1.T)

    SR_2 = np.linalg.inv(np.eye(Phi_1.shape[1]) - gamma_f * P_2.T)

    target_V = Phi_2 @ P_2.T @ SR_2  # @ np.linalg.inv(A @ SR_sym)


    return target_W,target_V,SR_1

def get_stationary_dist(P):

    v,V = np.linalg.eig(P.T)


    stat = V[:,np.argmax(v.real)]


    stat = stat.real

    return stat/(stat.sum())




