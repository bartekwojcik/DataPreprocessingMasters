import numpy as np
from typing import List, Tuple


class IrlProcessorResult:
    """
    Just a place holder
    """
    def __init__(
        self,
        weights,
        reward_matrix,
        policy,
        V,
        new_conversation,
        is_ok,
        list_of_t_W_intercept: List[
            Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ],
    ):
        self.list_of_t_W_intercept_policies_rewards = list_of_t_W_intercept
        self.V = V
        self.weights = weights
        self.reward_matrix = reward_matrix
        self.policy = policy
        self.new_conversation = new_conversation
        self.is_ok = is_ok
