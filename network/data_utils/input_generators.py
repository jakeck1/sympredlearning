from .state_iterators import DiscreteRandomWalkIterator
from .observation_mappers import DiscreteObservationMapper
import numpy as np

class InputGenerator:

    def __init__(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration

    max_iter = None


class IterableInputGenerator(InputGenerator):


    def __init__(self,iterable):
        super().__init__()

        self.iterator = iter(iterable)

        try:
            self.max_iter = len(self.iterator)
        except TypeError:
            pass

    def __next__(self):

        return next(self.iterator)



class StateSpaceInputGenerator(InputGenerator):

    def __init__(self,state_space_iterator,state_to_observation_mapper,return_state = False):

        #assert state_to_observation_mapper.check_compatibility(state_space_iterator.dummy_state), 'Dimensions of state space do not match'

        self.state_space_iterator = state_space_iterator

        self.state_to_observation_mapper = state_to_observation_mapper
        self.max_iter = self.state_space_iterator.max_iter
        self.return_state = return_state
    def __next__(self):

        state = next(self.state_space_iterator)
        observation = self.state_to_observation_mapper.sample(state)
        if self.return_state:
            return observation,state

        return observation

class StackedInputGenerators(InputGenerator):

    def __init__(self,generators):
        super().__init__()
        max_iters = np.array([gen.max_iter for gen in generators])
        try:

            self.max_iter = np.min(max_iters[max_iters!=None])
        except:
            self.max_iter = None
        self.generators = generators


    def __next__(self):

        return np.hstack([next(gen) for gen in self.generators])





def set_up_input_generator(P,Phi_1,Phi_2,normalize = True,return_state = False):
    rw_iterator = DiscreteRandomWalkIterator(P)

    if normalize:
        Phi_1 /= np.linalg.norm(Phi_1)
        Phi_2 /= np.linalg.norm(Phi_2)

    Phi = np.vstack((Phi_1, Phi_2))

    observation_mapper = DiscreteObservationMapper(Phi)

    input_generator = StateSpaceInputGenerator(rw_iterator, observation_mapper,return_state=return_state)

    return input_generator