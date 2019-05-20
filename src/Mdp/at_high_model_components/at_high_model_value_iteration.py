from typing import Tuple

from Mdp.at_high_model_components.at_high_model import AtHighMdpModel
import numpy as np


class AtHighValueIteration:
    def __init__(
        self, model: AtHighMdpModel, theta=0.0001, discount_factor=0.95
    ):
        self.theta = theta
        self.discount_factor = discount_factor
        self.G = model.graph
        self.n_s = len(model.states)
        self.n_a = len(model.actions)

    def get_optimal_policy(self, rewards: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates policy and V of states
        :param rewards: array of mappings state->rewards
        :return: Tuple of policy and Vs. A policy is a mapping where policy[state]->action where state index is given state
        0 - None
        1 - H at L
        2 - L at H
        3 - Mutual
        """
        return self.__calculate_value_iteration(rewards)

    def __calculate_value_iteration(self, rewards: np.ndarray):

        V = np.zeros(self.n_s)
        policy = np.zeros((self.n_s,))

        while True:
            delta = 0

            for s in range(self.n_s):

                action_values = np.zeros(self.n_a)
                for a in range(self.n_a):
                    for prob, next_state in self.G[s][a]:
                        # the reward is the reward of given next_state
                        reward = rewards[next_state]
                        action_values[a] += prob * (reward + self.discount_factor * V[next_state])
                best_a = np.max(action_values)

                delta = max(delta, np.abs(best_a - V[s]))
                V[s] = best_a

            if delta < self.theta:
                break

        for s in range(self.n_s):
            action_values = np.zeros(self.n_a)
            for a in range(self.n_a):
                for prob, next_state in self.G[s][a]:
                    # the reward is the reward of given next_state
                    reward = rewards[next_state]
                    action_values[a] += prob * (
                        reward + self.discount_factor * V[next_state]
                    )
            best_a = np.argmax(action_values)
            policy[s] = best_a

        return policy, V
