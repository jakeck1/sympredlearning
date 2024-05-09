from typing import Dict
import neuronav.utils as utils
import enum
import copy
import numpy as np
from gym import spaces
from neuronav.envs.graph_env import GraphEnv
from neuronav.envs.grid_env import GridEnv,GridTemplate
import networkx as nx

def create_grid_env(env_name,**kwargs):


    grid_env = GridEnv(GridTemplate[env_name])

    return GridEnvWrapper(grid_env)

def create_circular_tree_graph(n_leafs):

    G = nx.DiGraph()

    c = 1
    last_nodes = [0]
    for j in range(len(n_leafs)):
        new_nodes = []

        current_n_leafs = n_leafs[j]
        for last_node in last_nodes:
           # print(last_node,'last node')
            for i in range(current_n_leafs):
                new_node = c
                G.add_edge(last_node,new_node)
                new_nodes.append(new_node)
                c+=1

        last_nodes = new_nodes

    for last_node in last_nodes:
        G.add_edge(last_node,0)

    return GraphEnvWrapper(G)






class GridEnvWrapper:


    def __init__(self,grid_env):



        self._grid_env = grid_env

        self.T = None

        self.G = self.construct_stategraph()

    def __getattr__(self,attr):
        return getattr(self._grid_env,attr)

    def get_transition_model(self,only_feasible_states =True):

        if self.T is not None:
            return self.T.copy()

        else:

            action_size = self.action_space.n
            state_size = self.state_size
            T = np.zeros([action_size, state_size, state_size])

            for state in range(state_size):
                pos = self.state_to_gridpos(state)
                for action in range(action_size):
                    direction = self.direction_map[action]
                    new_pos = pos + direction
                    if self.check_target(new_pos):

                        new_state = self.gridpos_to_state(new_pos[0], new_pos[1])
                        T[action, state, new_state] = 1.0

                    elif only_feasible_states:
                        if self.check_target(pos):
                            T[action, state, state] = 1.0
                    else:
                        T[action, state, state] = 1.0
            self.T = T
            return T.copy()
    def state_to_gridpos(self,r):
        '''
        Invert the map (i,j) -> im + j
        to get grid position from integer state
        j  = r mod m
        i =  (r-j)/m
        works only for square grids
        '''
        j = r % self.grid_size
        i = (r-j)//self.grid_size

        return (i,j)


    def gridpos_to_state(self,i,j):

        return (i*self.grid_size)+j

    def construct_stategraph(self,add_self_edges=True,only_feasible_states=True):
        obs = self.reset()
        grid = self.symbolic_obs()
        valid_spots = grid[:, :, 4] == 0


        G = self.array_representation_to_graph(valid_spots, add_self_edges=add_self_edges,
                                                 only_feasible_states=only_feasible_states)
        return G
    def array_representation_to_graph(self,arr, add_self_edges=False, only_feasible_states=True):
        '''
        Turns an array representation of a nagivation environment to a Graph

            args:
            -----
            arr - np.array, shape (n,m)
                  A numpy array representing an nxm rectangular environment. Entries with value 1 are valid states of the
                  environment, while entries with value 0 correspond to invalid states, e.g. barriers
            add_self_edges - boolean
                  if true, whenever there is an invalid transition, a self-edge is added instead
            returns:
            -----
            G - DistanceAwareGraph
                A graph representing all valid states as nodes, where edges exist between adjacent valid states
        '''
        G = nx.DiGraph()
        n, m = arr.shape

        for i in range(n):
            for j in range(m):

                if arr[i, j] == 0 and (not only_feasible_states):
                    current_node = (i,j)
                    G.add_node(current_node)
                elif (arr[i, j] == 1):
                    current_node = (i,j)
                    G.add_node(current_node)
                    w = 0  # Initialize weight for potential self-edge
                    add_self_edge = False  # Initialize flag for self-edge addition

                    # Check and add an edge for the left neighbor
                    if i > 0:
                        if arr[i - 1, j] == 1:
                            G.add_edge(current_node, (i - 1, j), weight=1)
                    elif add_self_edges:
                        add_self_edge = True
                        w += 1
                    if i < n - 1:
                        if arr[i + 1, j] == 1:
                            G.add_edge(current_node, (i + 1, j), weight=1)
                    elif add_self_edges:
                        add_self_edge = True
                        w += 1

                    # Check and add an edge for the upper neighbor
                    if j > 0:
                        if arr[i, j - 1] == 1:
                            G.add_edge(current_node, (i, j - 1), weight=1)
                    elif add_self_edges:
                        add_self_edge = True
                        w += 1

                    if j < m - 1:
                        if arr[i, j + 1] == 1:
                            G.add_edge(current_node, (i, j + 1), weight=1)
                    elif add_self_edges:
                        add_self_edge = True
                        w += 1

                    # Add a self-edge if necessary
                    if add_self_edge and add_self_edges:
                        G.add_edge(current_node, current_node, weight=w)

        return DistanceAwareGraph(G)



    def get_objects(self,reward_location,reset = True):
        if reset:
            _ = self.reset()

        objects = self.objects
        new_objects = {}

        for obj_key in objects.keys():
            new_objects[obj_key] = objects[obj_key]

        new_objects['rewards'] = {}

        new_objects['rewards'][tuple(reward_location)] = 1.0

        return new_objects


    def get_unit_reward_vector(self,target,grid_pos=True):

        w = np.zeros(self.state_size)
        if grid_pos:
            target_state = self.gridpos_to_state(target[0],target[1])
        else:
            target_state = target
        w[target_state]=1.0

        return w

    def distance(self,source,target):
        #print(source,target)
        return self.G.distance(source,target)


class DistanceAwareGraph:


    def __init__(self,graph):

        self._graph = graph
        self.distances = dict(nx.all_pairs_shortest_path_length(graph))
    def __getattr__(self, item):
        return getattr(self._graph,item)

    def distance(self,source,target):

        #neuronav gridenv stores positions as lists, need tuples for hashable node identifiers
        if isinstance(source,list):
            source = tuple(source)
        if isinstance(target,list):
            target = tuple(target)
        distance =  self.distances.get(source,{}).get(target,None)

        if not distance:
            raise ValueError('It seems like the combination (source,target) is not a valid path in the graph')

        return distance

    def mean_distance(self):

        nodes = list(self.nodes)

        ds = []
        for source in nodes:
            for target in nodes:
                if source!=target:
                    ds.append(self.distance(source,target))

        return np.mean(ds)






class GraphObservation(enum.Enum):
    onehot = "onehot"
    index = "index"
    images = "images"


class GraphEnvWrapper(GraphEnv):

    def __init__(
        self,
        G:  nx.DiGraph,
        objects: Dict = None,
        obs_type: GraphObservation = GraphObservation.index,
        seed: int = None,
        use_noop: bool = False,
        torch_obs: bool = False,
        start_pos: int = 0,
        relabel_nodes = True
    ):
        self.use_noop = use_noop
        self.rng = np.random.RandomState(seed)

        # Convert input strings to corresponding enums

        if isinstance(obs_type, str):
            obs_type = GraphObservation(obs_type)

        # Generate layout
        if relabel_nodes:
            # need nodes to be integer-named for some of the methods
            nodes_list = list(G.nodes())
            mapping = {node: index for index, node in enumerate(nodes_list)}
            G = nx.relabel_nodes(G, mapping)
        G = DistanceAwareGraph(G)

        self.G = G
        self.generate_layout(objects,start_pos)

        self.running = False
        self.obs_mode = obs_type
        print(self.obs_mode,'obsmode')
        self.torch_obs = torch_obs
        self.base_objects = {"rewards": {}}
        self.T = None
        if obs_type == GraphObservation.onehot:
            self.obs_space = spaces.Box(
                low=0, high=1, shape=(self.state_size,), dtype=np.int32
            )
        elif obs_type == GraphObservation.index:
            self.obs_space = spaces.Box(
                low=0, high=self.state_size - 1, shape=(1,), dtype=np.int32
            )
        elif obs_type == GraphObservation.images:
            self.obs_space = spaces.Box(0, 1, shape=(32, 32, 3))
            self.images = utils.cifar10()[0]

    def generate_layout(self,  objects, start_pos):
        degrees = [tup[1] for tup in list(self.G.out_degree())]
        action_size = max(degrees)
        self.template_objects = objects
        self.agent_start_pos = start_pos
        self.action_space = spaces.Discrete(action_size + self.use_noop)
        self.state_size = len(degrees)

    def get_transition_model(self):

        if self.T is not None:
            return self.T.copy()

        T = np.zeros((self.action_space.n, self.state_size, self.state_size))

        nodelist = list(self.G.nodes)

        for i in range(self.state_size):
            node = nodelist[i]
            edges = list(self.edges(node))
            n_edges = len(edges)

            diff = max(self.action_space.n - n_edges - self.use_noop, 0)

            js = []
            for k in range(len(edges)):
                v = edges[k][1]
                j = nodelist.index(v)
                T[k, i, j] = 1.0
                js.append(j)

            # we randomly sample among edges for the remaining actions
            for l in range(diff):
                for j in js:
                    T[l + n_edges, i, j] = 1.0 #/ len(js)

                T[l+n_edges,i,:]/=T[l+n_edges,i,:].sum()
            if self.use_noop:
                T[self.action_space.n - 1, i, i] = 1.0
        self.T = T


        return T.copy()
    def step(self, action: int):
        """
        Takes a step in the environment given an action.
        """
        if not self.running:
            print(
                f"Please call {self.__class__.__name__}.reset() before {self.__class__.__name__}.step()."
            )
            return None, None, None, None
        if self.done:
            print(
                f"Episode finished. Please reset the {self.__class__.__name__} environment."
            )
            return None, None, None, None

        # No-op action
        if self.use_noop and action == self.action_space.n - 1:
            reward = 0
        else:
            # Stochastic action selection
            if self.stochasticity > self.rng.rand():
                action = self.rng.randint(0, self.action_space.n)

            candidate_position = self.rng.choice(np.arange(self.state_size), p=self.T[action,self.agent_pos])


            self.agent_pos = candidate_position
            reward = 0
            if self.agent_pos in self.objects["rewards"]:
                reward += self.objects["rewards"][self.agent_pos]
                self.done = True

            reward -= self.time_penalty


            node = list(self.G.nodes)[self.agent_pos]
            if len(self.edges(node)) == 0:
                self.done = True

        return self.observation, reward, self.done, {}


    @property
    def edges(self):
        return self.G.edges

    def get_objects(self,reward_location,reset = True):
        if reset:
            _ = self.reset()

        objects = self.objects
        new_objects = {}

        for obj_key in objects.keys():
            new_objects[obj_key] = objects[obj_key]

        new_objects['rewards'] = {}

        new_objects['rewards'][reward_location] = 1.0

        return new_objects


    def get_unit_reward_vector(self,target):

        w = np.zeros(self.state_size)
        w[target]=1.0

        return w

    def distance(self,source,target):
        return self.G.distance(source,target)
    def get_observation(self):
        """
        Returns an observation corresponding to the current state.
        """
        if self.obs_mode == GraphObservation.onehot:
            return utils.onehot(self.agent_pos, self.state_size)
        elif self.obs_mode == GraphObservation.index:
            return self.agent_pos
        elif self.obs_mode == GraphObservation.images:
            return np.rot90(self.images[self.agent_pos], k=3)
        else:
            raise NotImplementedError
    def reset(
        self,
        agent_pos: int = None,
        objects: Dict = None,
        random_start: bool = False,
        time_penalty: float = 0.0,
        stochasticity: float = 0.0,
    ):
        """
        Resets the environment to initial configuration.
        """
        self.running = True
        self.stochasticity = stochasticity
        self.time_penalty = time_penalty
        if agent_pos != None:
            self.agent_pos = agent_pos
        elif random_start:
            self.agent_pos = self.get_free_spot()
        else:
            self.agent_pos = self.agent_start_pos
        self.done = False
        if objects != None:
            use_objects = copy.deepcopy(self.base_objects)
            for key in objects.keys():
                if key in use_objects.keys():
                    use_objects[key] = objects[key]
            self.objects = use_objects
        else:
            self.objects = self.base_objects
        return self.observation
