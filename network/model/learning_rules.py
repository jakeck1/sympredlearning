import numpy as np

class LearningRule:

    def __init__(self):
        pass

    def weight_update(self, W, post_new, post_old, pre_new, pre_old):
        raise NotImplementedError


class NormalizingLearningRule(LearningRule):

    def __init__(self, alpha=1.0, beta=0.0):
        self.alpha = alpha
        self.beta = beta

    def weight_update(self, W, post_new, post_old, pre_new, pre_old):
        pre_post = np.outer((post_new - W @ pre_old), pre_old)
        post_pre = np.outer((post_old - W @ pre_new), pre_new)
        # Return the learning rule
        return self.alpha * pre_post + self.beta * post_pre


class NormalizingLearningRuleWithKernels(LearningRule):

    def __init__(self, alpha=1.0, beta=0.0, pre_decay=1.0, post_decay=1.0):
        self.alpha = alpha
        self.beta = beta
        self.post_trace = 0.0
        self.pre_trace = 0.0
        self.pre_decay = pre_decay
        self.post_decay = post_decay

    def weight_update(self, W, post_new, post_old, pre_new, pre_old):
        self.post_trace = (1 - self.post_decay) * self.post_trace + self.post_decay * post_new
        self.pre_trace = (1 - self.pre_decay) * self.pre_trace + self.pre_decay * pre_new

        pre_post = np.outer(post_new, self.pre_trace) - np.outer(W @ self.pre_trace, self.pre_trace)

        post_pre = np.outer(self.post_trace, pre_new) - np.outer(W @ pre_new, pre_new)

        # Return the learning rule
        return self.alpha * pre_post + self.beta * post_pre


class EmptyLearningRule(LearningRule):

    def weight_update(self, W, post_new, post_old, pre_new, pre_old):
        return 0.0
