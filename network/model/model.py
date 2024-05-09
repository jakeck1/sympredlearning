import numpy as np
from .layer import SingleSRLayer



class MultilayerSRNetwork:

    def __init__(self, layers):

        if isinstance(layers, SingleSRLayer):
            layers = [layers]
        self.layers = layers

        self.n_cells = [self[i].n_cells for i in range(len(self))]

        self.split_indices = np.cumsum(self.n_cells[:-1])


    def update(self, inputs, dt, learning_rates=1.0, update_weights = True,update_scheme = 'sequential'):


        if isinstance(learning_rates,float):
            learning_rates = [learning_rates for i in range(len(self))]

        split_inputs = np.split(inputs,self.split_indices)

        old_activities = [self[i].activity.copy() for i in range(len(self))]


        top_down = [self[i].backward() for i in range(len(self))]

        new_activities = []

        if update_scheme == 'sequential':

            for i in range(len(self)):
                if i == 0:
                    bottom_up_inp = 0.0
                else:
                    # bottom up input gets taken from the previous layer, which has already been updated
                    bottom_up_inp = self[i - 1].forward()

                if i == (len(self) - 1):
                    top_down_inp = 0.0
                else:
                    top_down_inp = top_down[i + 1]
                lateral_inp = split_inputs[i]

                # this updates activity of this layer in place
                new_activities.append(self[i].update(dt, lateral_inp, bottom_up_inp, top_down_inp))

                if update_weights:
                    if i > 0:

                        self[i].backward_weights.update(learning_rates[i] * dt, new_activities[i - 1],
                                                        old_activities[i - 1], new_activities[i], old_activities[i])

                        self[i - 1].forward_weights.update(learning_rates[i] * dt, new_activities[i], old_activities[i],
                                                           new_activities[i - 1], old_activities[i - 1])

                    self[i].recurrent_weights.update(learning_rates[i] * dt, new_activities[i], old_activities[i],
                                                     new_activities[i], old_activities[i])
        elif update_scheme == 'concurrent':
            bottom_up = [self[i].forward() for i in range(len(self))]

            activity_updates = []
            for i in range(len(self)):
                if i==0:
                    bottom_up_inp = 0.0
                else:
                    bottom_up_inp= bottom_up[i-1]
                if i == (len(self)-1):
                    top_down_inp = 0.0
                else:
                    top_down_inp = top_down[i+1]
                lateral_inp = split_inputs[i]

                new_activities.append(self[i].update(dt, lateral_inp, bottom_up_inp, top_down_inp))

                if update_weights:
                    if i>0:
                        self[i].backward_weights.update(learning_rates[i]*dt,new_activities[i-1],old_activities[i-1],new_activities[i],old_activities[i])

                        self[i-1].forward_weights.update(learning_rates[i]*dt,new_activities[i],old_activities[i],new_activities[i-1],old_activities[i-1])

                    self[i].recurrent_weights.update(learning_rates[i]*dt,new_activities[i],old_activities[i],new_activities[i],old_activities[i])

        return new_activities

    def __getitem__(self,i):
        return self.layers[i]

    def __len__(self):
        return len(self.layers)



