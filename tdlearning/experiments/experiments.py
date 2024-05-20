
from ..utils.utils import run_episode,run_copy_episode
import numpy as np
from tdlearning.envs.env_wrappers import *
from tdlearning.agents.SRAgent import TDSR_WM



def run_experiment(env_name,mode='fixed_time',lr=1e-1,gamma=0.7,seed=0.0,num_steps=120,num_episodes=200,normalize=False,provide_transition_model=True):
    
    
    np.random.seed(seed)  
    

    if env_name == 'tree':

        leafs = [4,2,2]
        env = create_circular_tree_graph(leafs)
    else:
        env = create_grid_env(env_name)
    T = None
    if provide_transition_model:
        T = env.get_transition_model()
    asym_agent = TDSR_WM(env.state_size, env.action_space.n, lr=lr, forward_weight=1.0,
                         backward_weight=0.0, gamma=gamma, poltype='softmax',T=T)
    sym_agent = TDSR_WM(env.state_size, env.action_space.n, lr=lr, forward_weight=0.5,
                        backward_weight=0.5, gamma=gamma, poltype='softmax',T=T)
    
    if mode == 'fixed_time':
        steps_asym = run_single_agent_generalization_experiment(env, asym_agent, num_steps=num_steps, num_episodes=num_episodes)
        steps_sym = run_single_agent_generalization_experiment(env, sym_agent, num_steps=num_steps, num_episodes=num_episodes)
    
    elif mode =='performance':
        steps_asym=run_single_agent_generalization_experiment_with_performance_criterion(env,asym_agent,num_steps=num_steps)


        steps_sym=run_single_agent_generalization_experiment_with_performance_criterion(env,sym_agent,num_steps=num_steps)

    elif mode =='copy':
        steps_asym,steps_sym = run_copy_agent_generalization_experiment(env,asym_agent,sym_agent,num_steps=num_steps,num_episodes=num_episodes,normalize=normalize)

            
    return steps_sym, steps_asym




def run_single_agent_generalization_experiment(env,agent,num_steps=150,num_episodes=200,provide_model_from_start=True,train_target = None,test_target = None):


            obs = env.reset()

            episode_lens = []

            if provide_model_from_start:
                T = env.get_transition_model()
                agent.T = T.copy()

            if train_target is None:
                train_target = env.get_free_spot()
            if test_target is None:
                #sample a test target different from train target
                while test_target in [None,train_target]:
                    test_target = env.get_free_spot()

            train_objects = env.get_objects(train_target)
            test_objects = env.get_objects(test_target)


            for i in range(num_episodes):
                #print(i)
                agent, steps, total_reward = run_episode(env, agent, num_steps,objects = train_objects, random_start=True)
                episode_lens.append(steps)
            obs = env.reset()
            agent.learn_SR = False
            for i in range(num_episodes):
                agent, steps, total_reward = run_episode(env, agent, num_steps, objects=test_objects,
                                                         random_start=True)
                episode_lens.append(steps)
            episode_lens = np.array(episode_lens)


            return episode_lens


def run_single_agent_generalization_experiment_with_performance_criterion(env, agent,performance_criterion=2, steps_good_performance=8, num_steps=200,
                                                           provide_model_from_start=True, train_target=None,
                                                           test_target=None,max_episodes=1000):

            obs = env.reset()

            performances = []

            if provide_model_from_start:
                T = env.get_transition_model()
                agent.T = T.copy()

            if train_target is None:
                train_target = env.get_free_spot()
            if test_target is None:
                #sample a test target different from train target
                while test_target in [None,train_target]:
                    test_target = env.get_free_spot()

            train_objects = env.get_objects(train_target)
            test_objects = env.get_objects(test_target)
            good_performance = False
            start_pos = train_target
            
            
            
            c=0
            while (not good_performance) and(c<max_episodes) :
                while start_pos == train_target:
                    start_pos = env.get_free_spot()

                agent, steps, total_reward = run_episode(env, agent, num_steps, objects=train_objects,
                                                         start_pos=start_pos)
                performance = steps - env.distance(start_pos,train_target)
                performances.append(performance)
                if len(performances) > steps_good_performance:
                    if np.mean(performances[-steps_good_performance:]) < performance_criterion:
                        good_performance = True
                        
                c+=1
               
            agent.learn_SR = False

            #we provide info about new reward location immediately
            w = env.get_unit_reward_vector(test_target)
            agent.w = w

            performances = []
            for start_pos in env.make_free_spots():

                if start_pos == test_target:
                    continue
                agent, steps, total_reward = run_episode(env, agent, num_steps, objects=test_objects,
                                                         start_pos=start_pos)
                performance = steps - env.distance(start_pos,test_target)

                performances.append(performance)

            return performances




def run_copy_agent_generalization_experiment(env,base_agent,copy_agent,num_episodes=200,num_steps=150,provide_model_from_start=True,train_target = None,test_target = None,normalize=True):
    obs = env.reset()

    episode_lens_base = []
    episode_lens_copy =[]

    if provide_model_from_start:
        T = env.get_transition_model()
        base_agent.T = T.copy()
        copy_agent.T = T.copy()

    if train_target is None:
        train_target = env.get_free_spot()
    if test_target is None:
        # sample a test target different from train target
        while test_target in [None, train_target]:
            test_target = env.get_free_spot()

    train_objects = env.get_objects(train_target)
    test_objects = env.get_objects(test_target)

    for i in range(num_episodes):
        base_agent,copy_agent, steps, total_reward = run_copy_episode(env, base_agent,copy_agent, num_steps, objects=train_objects, random_start=True,normalize_update=normalize)
        episode_lens_base.append(steps)
        episode_lens_copy.append(steps)
    obs = env.reset()
    base_agent.learn_SR = False
    copy_agent.learn_SR = False
    for i in range(num_episodes):
        base_agent, steps, total_reward = run_episode(env, base_agent, num_steps, objects=test_objects,
                                                 random_start=True)


        episode_lens_base.append(steps)

        copy_agent, steps, total_reward = run_episode(env, copy_agent, num_steps, objects=test_objects,
                                                      random_start=True)

        episode_lens_copy.append(steps)


    return np.array(episode_lens_base),np.array(episode_lens_copy)


