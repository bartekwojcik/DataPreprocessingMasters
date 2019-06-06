import numpy as np
from typing import List, Tuple

from inverse_reinforcement_learning.feature_expectations_extractor import FeatureExpectationExtractor


class RewardCalculator:

    def __init__(self, reward_shape:tuple, states:List[Tuple]):
        self.states = states
        self.reward_shape = reward_shape

    def calculate_reward(self, W:np.ndarray, intercept)-> np.ndarray:
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
            current_state = self.states[s]
            ##this is just to make work around zero vector problem
            state_vector = FeatureExpectationExtractor.calculate_state_vector(current_state,self.states)
            R[s] = np.dot(W.T, state_vector) + intercept

        return R


