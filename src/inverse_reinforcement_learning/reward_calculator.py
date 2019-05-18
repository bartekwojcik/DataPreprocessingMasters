from Mdp.mdp_utils import MdpUtils
from Mdp.models.simple_16_deterministic_model import Simple16ActionMdpModel
import numpy as np

class RewardCalculator:

    def __init__(self, n_states:int):
        self.n_states = n_states
        self.eye_states = np.eye(self.n_states)

    def calculate_reward(self, W:np.ndarray)-> np.ndarray:
        """
        Calculates reward for each state given array of Weights
        http://3dvision.princeton.edu/courses/COS598/2014sp/slides/lecture07_reinforcement.pdf
        :param W: weights
        :return: array of reward of dim [States,Actions]
        """

        R = np.zeros((self.n_states,))

        for s in range(self.n_states):
            state_vector = self.eye_states[s]
            R[s] = np.dot(W.T, state_vector)

        return R


