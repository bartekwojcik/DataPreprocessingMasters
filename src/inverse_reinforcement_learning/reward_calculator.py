import numpy as np

class RewardCalculator:

    def __init__(self, reward_shape:tuple, states:np.ndarray):
        self.states = states
        self.reward_shape = reward_shape

    def calculate_reward(self, W:np.ndarray)-> np.ndarray:
        """
        Calculates reward for each state given array of Weights
        http://3dvision.princeton.edu/courses/COS598/2014sp/slides/lecture07_reinforcement.pdf
        :param W: weights
        :return: array of reward of dim [States,Actions]
        """

        shape_length = self.reward_shape[0]
        #shape of 16, because reward PER state
        R = np.zeros((shape_length,))

        for s in range(shape_length):
            state_vector = self.states[s]
            ##this is just to make work around zero vector problem
            if np.all(state_vector==0):
                R[s] = -8
            else:
                R[s] = np.dot(W.T, state_vector)

        return R


