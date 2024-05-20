from network.data_utils.run_episode import run_episode
from scipy.linalg import circulant
from network.model.utils import *
from network.data_utils.input_generators import set_up_input_generator
from network.data_utils.lr_scheduler import *
from network.model.weights import NormalInitializer,AbsTransform,PowerTransform,PowerScaler,UniformInitializer
import os
import numpy as np


def run_convergence_experiment(
        n_iterations: int=1,
        n_saves: int = int(1e5),
        dt: float = 0.001,
        lr =1.0,
        n_cells_layer_1: int = 40,
        n_cells_layer_2: int = 40,
        tau_learning: float = 0.12,
        tau_learning_f: float =0.12,
        tau_activity: float = 0.001,
        gamma_r: float = 0.7,
        gamma_f: float =0.7,
        alpha_1 = 0.5,
        beta_1 = 0.5,
        alpha_2 = 1.0,
        beta_2 = 0.0,
        activation_functions = (None,None),
        n_states: int = 30,
        P = None,
        circular_track = True,
        random_transition_matrix =False,
        sample_random_transition_matrix: str ='gaussian',
        circular_rw_params = (0,3,1.0),
        features = (None,None),
        feature_type: str = 'random_normal',
        non_negative_features=False,
        normalize_features = True,
        scale_initial_weights = -1,
        transform_weights_power = None,
        abs_transform_weights = False,
        loss_relative_to_first_step=False,
        only_final_loss = False
):


    if P is None:

        if circular_track:
            v = np.zeros(n_states)
            v[1] = 1

            (lower,upper,sigma) = circular_rw_params
            g = np.exp(-(np.linspace(lower, upper, n_states) ** 2) / sigma)
            v = np.convolve(g, v, 'same')
            v = np.roll(v, 1)
            P = circulant(v).T
            P /= P.sum(axis=1, keepdims=True)


        elif random_transition_matrix:
            if sample_random_transition_matrix == 'gaussian':
                P = np.abs(np.random.normal(size=(n_states, n_states)))
            elif sample_random_transition_matrix == 'uniform':
                P = np.abs(np.random.uniform(size=(n_states, n_states)))
            else:
                raise NotImplementedError

            P /= P.sum(axis=1, keepdims=True)


    n_states = P.shape[0]

    Phi_1,Phi_2 = features
    if Phi_1 is None:


        if feature_type == 'random_normal':
            Phi_1 = np.random.normal(size=(n_cells_layer_1, n_states))
        elif feature_type=='one_hot':
            Phi_1 = np.eye(n_states)
    if Phi_2 is None:

        if feature_type == 'random_normal':
            Phi_2 = np.random.normal(size=(n_cells_layer_2, n_states))
        elif feature_type == 'one_hot':
            Phi_2 = np.eye(n_states)

    if non_negative_features:
        Phi_1 = np.abs(Phi_1)
        Phi_2 = np.abs(Phi_2)

    if normalize_features:

        Phi_1 /= np.linalg.norm(Phi_1)
        Phi_2 /= np.linalg.norm(Phi_2)

    scaler = None
    if scale_initial_weights:

        scaler = PowerScaler(scale_initial_weights)

    transform = None
    if transform_weights_power:

        transform = PowerTransform(transform_weights_power)
    else:
        if abs_transform_weights:

             transform = AbsTransform()


    weight_initializer = NormalInitializer(scaler=scaler, transform=transform)

    input_generator = set_up_input_generator(P, Phi_1, Phi_2)

    model = create_model((n_cells_layer_1, n_cells_layer_2), ((alpha_1, beta_1), (alpha_2, beta_2)), tau_learning, tau_activity,
                         gamma_r=gamma_r,gamma_f=gamma_f, tau_learning_f=tau_learning_f, weight_initializer_W=weight_initializer,
                         weight_initializer_V=weight_initializer, activation_functions=activation_functions)

    losses_W = []
    losses_V = []
    A,B,Q,T,gamma_r = quantities_for_convergence_diff(model[0].recurrent_weights.learning_rule.alpha,
                                           model[0].recurrent_weights.learning_rule.beta,
                                           model[0].forward_weights.learning_rule.alpha,
                                           model[0].forward_weights.learning_rule.beta,
                                           P, Phi_1, Phi_2, model[0].gamma_r, model[1].gamma_f)

    if loss_relative_to_first_step:
        diff_W_0, diff_V_0 = conv_diff(model[0].W_r, model[0].W_f, A,B,Q,T,gamma_r)
        loss_W_0 = np.linalg.norm(diff_W_0)
        loss_V_0 = np.linalg.norm(diff_V_0)

    for i in range(n_saves):
        out = run_episode(model, input_generator, dt,lr = lr,iterations=n_iterations)
        if not only_final_loss:
            diff_W,diff_V = conv_diff(model[0].W_r,model[0].W_f,A,B,Q,T,gamma_r)

            loss_W = np.linalg.norm(diff_W)
            loss_V = np.linalg.norm(diff_V)
            if loss_relative_to_first_step:
                loss_W/=loss_W_0
                loss_V/=loss_V_0



            losses_W.append(loss_W)
            losses_V.append(loss_V)

    if only_final_loss:
            diff_W,diff_V = conv_diff(model[0].W_r,model[0].W_f,A,B,Q,T,gamma_r)

            loss_W = np.linalg.norm(diff_W)
            loss_V = np.linalg.norm(diff_V)
            if loss_relative_to_first_step:
                loss_W/=loss_W_0
                loss_V/=loss_V_0
        
            return loss_W,loss_V

    else:

        return np.array(losses_W),np.array(losses_V)



def sample_features(n_features, n, sigma=1.0):
    out = []
    x = np.linspace(-3, 3, n)
    for i in range(n_features):
        mu = np.random.uniform(-3, 3)

        f = np.exp((-(x - mu) ** 2) / sigma)
        out.append(f)

    return np.array(out)


def get_mean_firingrate_in_states(firing_rates, states, n_states):
    out = np.zeros((firing_rates.shape[1], n_states))

    for state in states:
        out[:, state] = firing_rates[states == state].mean(axis=0)

    return out


def calculate_com(firing_rates, states, distances):
    mean_firing_rates = get_mean_firingrate_in_states(firing_rates, states, len(np.unique(states)))

    com = mean_firing_rates @ distances
    com /= mean_firing_rates.sum(axis=1)

    return com




def shift_experiment(
        n_reps  =30,
        n_cells_layer_1: int = 100,
        n_cells_layer_2: int = 100,
        tau_learning: float = 0.12,
        tau_learning_f: float =0.12,
        tau_activity: float = 0.001,
        gamma_r: float = 0.7,
        gamma_f: float =0.7,
        alpha_1 = 0.5,
        beta_1 = 0.5,
        alpha_2 = 1.0,
        beta_2 = 0.0,
        activation_functions = (None,None),
        n_states = 50,
        P  = None,
        vel = 15,
        dt: float = None,
        lr =0.005,
        len_track=300,
        n_laps = 25,
        sigma_features = 0.1,
        sigma_noise=0.01,
        cooldown =10,
        path_to_save=None,
        experiment_id='com'):



        if P is None:
            v = np.zeros(n_states)
            v[1] = 1.0
            A_right = circulant(v).T
            A_left = circulant(v)
            I = np.eye(n_states)
            P = 0.9 * A_right + 0.0 * A_left + 0.1 * I

            P[n_states - 1, :] = 0.0
            P[n_states - 1, n_states - 1] = 1.0

            P /= P.sum(axis=1, keepdims=True)
        n_states = P.shape[0]

        if dt is None:
            dt = (len_track / n_states) / vel

        distances = np.linspace(0, len_track, n_states)




        s_1 = []
        s_2 = []

        for _ in range(n_reps):

            Phi_1 = sample_features(n_cells_layer_1, n_states, sigma=sigma_features)
            Phi_2 = sample_features(n_cells_layer_2, n_states, sigma=sigma_features)

            Phi_1 += np.abs(np.random.normal(scale=sigma_noise,size=Phi_1.shape))
            Phi_2 += np.abs(np.random.normal(scale=sigma_noise,size=Phi_2.shape))

            #Phi_1 /= np.linalg.norm(Phi_1)
            #Phi_2 /= np.linalg.norm(Phi_2)


            input_generator = set_up_input_generator(P, Phi_1, Phi_2, return_state=True,normalize=False)

            model = create_model((n_cells_layer_1, n_cells_layer_2), ((alpha_1,beta_1), (alpha_2,beta_2)), tau_learning, tau_activity,
                                 gamma_r,gamma_f=gamma_f, tau_learning_f=tau_learning_f,activation_functions=activation_functions)

            ac_1 = []
            ac_2 = []
            stat = []

            n_rest_at_end = 1
            for l in range(n_laps):

                activities_1 = []
                activities_2 = []
                states = []
                c = 0

                #print('lap ', l)
                state = 0

                activities_1.append(model[0].activity.copy())
                activities_2.append(model[1].activity.copy())
                states.append(state)

                input_generator.state_space_iterator.state = 0
                while c < n_rest_at_end:
                    inp, state = next(input_generator)
                   # print(state)

                    model.update(inp, dt, lr)

                    activities_1.append(model[0].activity.copy())
                    activities_2.append(model[1].activity.copy())
                    states.append(state)
                    if state == n_states - 1:
                        c += 1
                for i in range(cooldown):
                    #let the model run with zero input for a few steps at end of lap
                    inp = np.zeros(n_cells_layer_1 + n_cells_layer_2)
                    model.update(inp, dt, lr)
                ac_1.append(np.array(activities_1))
                ac_2.append(np.array(activities_2))
                stat.append(np.array(states))


            l_1 = np.zeros((n_laps, n_cells_layer_1))
            l_2 = np.zeros((n_laps, n_cells_layer_2))

            middle_lap = n_laps // 2
            com_1_m = calculate_com(ac_1[middle_lap], stat[middle_lap], distances)
            com_2_m = calculate_com(ac_2[middle_lap], stat[middle_lap], distances)

            for i in range(n_laps):
                com_1 = calculate_com(ac_1[i], stat[i], distances)
                com_2 = calculate_com(ac_2[i], stat[i], distances)

                l_1[i] = com_1 - com_1_m
                l_2[i] = com_2 - com_2_m

            s_1.append(l_1)
            s_2.append(l_2)

        s_1 = np.array(s_1)
        s_2 = np.array(s_2)
        if path_to_save is not None:

            path_1 = os.path.join(path_to_save,'{}_first_layer.npy'.format(experiment_id))
            path_2 = os.path.join(path_to_save,'{}_second_layer.npy'.format(experiment_id))

            np.save(path_1, s_1)
            np.save(path_2, s_2)

        return s_1,s_2


def circular_rw_experiment(n_states=30,n_reps=30,dt=0.1):
    n_cells_layer_1 = n_states
    n_cells_layer_2 = n_states

    v = np.zeros(n_states)
    v[1] = 1.0
    A_right = circulant(v).T
    A_left = circulant(v)
    I = np.eye(n_states)

    P_uniform = A_right + A_left + I

    P_right = 0.5 * A_right + 0.25 * A_left + 0.25 * I

    P_left = 0.5 * A_left + 0.25 * A_right + 0.25 * I

    P_dict = {'uniform':P_uniform,'right':P_right,'left':P_left}

    walks = {}
    SRs={}

    U = np.eye(n_states)
    Phi_1 = U
    Phi_2 = U



    for kind,P in P_dict.items():

        P /= P.sum(axis=1, keepdims=True)

        input_generator = set_up_input_generator(P, Phi_1, Phi_2, return_state=True)
        model = create_model((n_cells_layer_1, n_cells_layer_2), ((0.5, 0.5), (1.0, 0.0)))
        states = []

        for i in range(int(60 // dt)):
            inp, state = next(input_generator)
            states.append(state)
        walks[kind]=np.array(states)

        n_reps = 30

        SR_1 = np.zeros((n_reps, n_states, n_states))
        SR_2 = np.zeros((n_reps, n_states, n_states))

        for q in range(n_reps):

            model = create_model((n_cells_layer_1, n_cells_layer_2), ((0.5, 0.5), (1.0, 0.0)),gamma_r=0.9)

            for i in range(int(1e5)):
                # train model to learn SR
                inp, state = next(input_generator)
                model.update(inp, dt=dt)

            #extract ST from model
            SR_1_a = np.zeros((n_states, n_states))
            SR_2_a = np.zeros((n_states, n_states))
            for k in range(n_states):
                inp = np.concatenate((Phi_1[:, k], Phi_2[:, k]))
                model.update(inp, dt=dt, update_weights=False)
                SR_1_a[k] = model[0].activity
                SR_2_a[k] = model[1].activity

            SR_1[q] = SR_1_a
            SR_2[q] = SR_2_a

        SRs[kind]=[SR_1,SR_2]

    return walks,SRs


