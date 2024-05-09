import numpy as np

class LearningRateScheduler:


   def __init__(self):
       self.max_iter = None

       pass

   def __iter__(self):
       return self
   def __next__(self):
       raise StopIteration


class ConstantLearningRateScheduler(LearningRateScheduler):


    def __init__(self,c):

        super().__init__()

        self.c = c

    def __next__(self):

        return self.c

class IterableLearningRateScheduler(LearningRateScheduler):


    def __init__(self,iterable):
        super().__init__()


        self.iterator = iter(iterable)
        try:
            self.max_iter = len(self.iterator)
        except TypeError:
            pass

    def __next__(self):

        return next(self.iterator)


class DecreasingLearningRateScheduler(LearningRateScheduler):



    def __init__(self):

        super().__init__()
        self.c = 1.0


    def __next__(self):

        out = 1/self.c
        self.c +=1.0

        return out



class DecayingLearningRateScheduler(LearningRateScheduler):



    def __init__(self,decay=0.99):

        super().__init__()
        self.c = 1.0
        self.decay = decay


    def __next__(self):

        out = self.c
        self.c *=self.decay

        return out

class CustomLearningRateScheduler(LearningRateScheduler):

    def __init__(self,start_decay = 1000):

        super().__init__()
        self.c = 1.0
        self.count = 1
        self.start_decay = start_decay
    def __next__(self):

        if self.count < self.start_decay:
            self.count +=1
            return 1.0
        else:

            out = 1/self.c
            self.c +=1.0
            return out



class SwitchLearningRateScheduler(LearningRateScheduler):

    def __init__(self,scheduler_1,scheduler_2,switch = 1000):

        super().__init__()
        self.schedulers = [scheduler_1,scheduler_2]
        self.c = 1.0
        self.count = 1
        self.switch = switch
    def __next__(self):

        if self.count < self.switch:
            self.count +=1
            return self.schedulers[0].__next__()
        else:

            return self.schedulers[1].__next__()


class MultipleLearningRateScheduler(LearningRateScheduler):


    def __init__(self,list_of_schedulers):

        self.schedulers = list_of_schedulers


    def __next__(self):

        return [scheduler.__next__() for scheduler in self.schedulers]