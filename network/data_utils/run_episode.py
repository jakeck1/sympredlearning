import numpy as np
from .input_generators import InputGenerator,IterableInputGenerator
from .lr_scheduler import LearningRateScheduler,IterableLearningRateScheduler,ConstantLearningRateScheduler


def store_no_states_rule(i):
    return False

def store_all_states_rule(i):
    return True

def store_random_states_rule(i,p):

    if np.random.random()>p:
        return True
    else:
        return False
def store_k_states_rule(i,k):
    if i%k ==0:
        return True
    else: return False




def run_episode(model,inp,dt,lr=1.0,iterations = None,store_states_rule = None):

    out = []

    if isinstance(inp,InputGenerator):
        input_generator = inp


    else:
        try: input_generator = IterableInputGenerator(inp)

        except: raise TypeError('input must either be an InputGenerator or an iterable')



    if isinstance(lr,(int,float)):

        scheduler = ConstantLearningRateScheduler(float(lr))

    elif isinstance(lr,LearningRateScheduler):
        scheduler = lr
    else:
        try:
            scheduler = IterableLearningRateScheduler(lr)
        except:
            raise TypeError('lr must be either numeric, an iterable or an instance of LearningRateScheduler')




    if iterations is None:
        max_iters = [input_generator.max_iter,scheduler.max_iter]

        iterations = np.min(max_iters[max_iters != None])

        if iterations is None:
            raise ValueError('Can only pass None for iterations if input or lr has finite length')

    if store_states_rule is None:
        store_states_rule = store_no_states_rule
    elif (store_states_rule== 'all') or (store_states_rule == True):
        store_states_rule = store_all_states_rule
    elif isinstance(store_states_rule,int):
        store_states_rule = lambda l: store_k_states_rule(l,store_states_rule)
    elif isinstance(store_states_rule,float):
        store_states_rule = lambda l: store_random_states_rule(l,store_states_rule)


    for i in range(iterations):

        learning_rate = next(scheduler)
        inp = next(input_generator)
        model.update(inp,dt = dt,learning_rates = learning_rate)

        if store_states_rule(i):
            out.append(model.states)


        #if np.any([np.all(np.isnan(layer.W_r)) for layer in model]):
          #  raise ValueError
    return out


