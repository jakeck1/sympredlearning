
import numpy as np
import os
def run_episode(
    env,
    agent,
    max_steps: int,
    start_pos=None,
    objects = None,
    random_start: bool = False,
    update_agent: bool = True,
    time_penalty: float = 0.0,
    collect_states: bool = False,
):
    """
    Performs a single episode of actions with the policy
    of a given agent in a given environment.
    """
    obs = env.reset(
        agent_pos=start_pos,
        objects=objects,
        random_start=random_start,
        time_penalty=time_penalty,
    )
    agent.reset()
    steps = 0
    episode_return = 0
    done = False
    if collect_states:
        states = []
    while not done and steps < max_steps:
        act = agent.sample_action(obs)
        obs_new, reward, done, _ = env.step(act)
        if update_agent:
            _ = agent.update([obs, act, obs_new, reward, done])
        if collect_states:
            states.append(obs)
        obs = obs_new
        steps += 1
        episode_return += reward
    if collect_states:
        return agent, steps, episode_return, states, done
    else:
        return agent, steps, episode_return




def run_copy_episode(env,
    agent,
    copy_agent,
    max_steps: int,
    start_pos=None,
    objects = None,
    random_start: bool = False,
    update_agent: bool = True,
    time_penalty: float = 0.0,
    collect_states: bool = False,
    normalize_update = False):
    obs = env.reset(
        agent_pos=start_pos,
        objects=objects,
        random_start=random_start,
        time_penalty=time_penalty,
    )

    agent.reset()
    copy_agent.reset()

    steps = 0

    episode_return = 0

    done = False

    if collect_states:
        states = []

    while not done and steps < max_steps:
        act = agent.sample_action(obs)
        obs_new, reward, done, _ = env.step(act)
        if update_agent:

            update = agent._update([obs, act, obs_new, reward, done],return_update = normalize_update)

            if normalize_update:
                normalize_update= np.linalg.norm(update)
            _ = copy_agent._update([obs, act, obs_new, reward, done],normalize_update=normalize_update)


        if collect_states:
            states.append(obs)
        obs = obs_new
        steps += 1
        episode_return += reward
    if collect_states:
        return agent, copy_agent, steps, episode_return, states, done
    else:
        return agent, copy_agent, steps, episode_return



def save_generalization_data(data,path,agents = None,suffices=None,makepath=True,overwrite=True):

    if agents is None:
        agents = ['symmetric','asymmetric']

    assert len(data)==len(agents), 'number of agent labels does not match number of arrays to save'

    if makepath:
        if not os.path.exists(path):
            os.makedirs(path)


    if isinstance(suffices,str):
        suffices = list(suffices)


    for i in range(len(data)):


        file_name = os.path.join(path,'gen_steps_{}'.format(agents[i]))
        if suffices is not None:
            for suffix in suffices:
                file_name += '_' + suffix
        file_name +='.npy'


        if (not overwrite) and (os.path.isfile(file_name)):
            print('file at {} already exists, set overwrite=True if you want to create data anyway'.format(file_name))

        else:
            np.save(file_name,data[i])
            print('saved data at {}'.format(file_name))




def load_generalization_data(path,agents = None,suffices=None):
    if agents is None:
        agents = ['symmetric','asymmetric']



    data = []
    for i in range(len(agents)):


        file_name = os.path.join(path,'gen_steps_{}'.format(agents[i]))
        if suffices is not None:
            for suffix in suffices:
                file_name += '_' + suffix
        file_name +='.npy'
        arr = np.load(file_name)
        data.append(arr)
        
    return data

