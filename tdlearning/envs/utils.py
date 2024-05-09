import networkx as nx
import numpy as np
from envs.graph_env_helpers import grid_env_to_graph
def helper_fun(r,m):
    '''
    Invert the map (i,j) -> im + j

    j  = r mod m
    i =  (r-j)/m

    works only for square grids
    '''

    j = r % m
    i = (r-j)//m

    return (i,j)


def invert_helper_fun(i,j,m):

    return (i*m)+j



def get_grid_env_transition_model(grid_env,only_feasible_states=True):


    action_size = grid_env.action_space.n
    state_size = grid_env.state_size
    T = np.zeros([action_size, state_size, state_size])

    for state in range(state_size):
        #print('state',state)
        pos = helper_fun(state,grid_env.grid_size)
       # print('pos',pos)
        for action in range(action_size):
          #  print('action',action)
            direction = grid_env.direction_map[action]

            new_pos = pos + direction
           # print('new pos',new_pos)
            if grid_env.check_target(new_pos):

                new_state = invert_helper_fun(new_pos[0],new_pos[1],grid_env.grid_size)
               # print(new_state,'new_state')
                T[action,state,new_state]=1.0

            elif only_feasible_states:
              if grid_env.check_target(pos):

                T[action,state,state]=1.0
            else:
                T[action, state, state] = 1.0

    return T


def get_optimal_policy(env, target_state):


    state_size = env.state_size
    action_size = env.action_space.n
    T = get_grid_env_transition_model(env, only_feasible_states=False)
    Pi = np.zeros((action_size,state_size))
    G = grid_env_to_graph(env,only_feasible_states=False)
    D = nx.floyd_warshall(G)
    target = invert_helper_fun(target_state[0],target_state[1],env.grid_size)

    for state in range(state_size):
        pos = helper_fun(state, env.grid_size)

        for action in range(action_size):
            direction = env.direction_map[action]

            new_pos = pos + direction
            if env.check_target(pos) and env.check_target(new_pos):
                new_state = invert_helper_fun(new_pos[0], new_pos[1], env.grid_size)

                if (D[state][target]>0) and (D[new_state][target]<D[state][target]):


                    Pi[action,state]=np.random.uniform()

                elif D[state][target]==0:
                    Pi[action,state]=1/action_size

            else:
                Pi[action,state]=1/action_size


    Pi/=Pi.sum(axis=0,keepdims=True)

    return Pi,T











