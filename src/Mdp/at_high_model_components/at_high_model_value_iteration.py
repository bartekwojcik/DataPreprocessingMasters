from typing import Tuple

import settings
from Mdp.at_high_model_components.at_high_model import AtHighMdpModel
import numpy as np


class AtHighValueIteration:
    def __init__(
        self, model: AtHighMdpModel, theta= settings.POLICY_THETA, discount_factor= settings.DISCOUNT_FACTOR
    ):
        self.theta = theta
        self.discount_factor = discount_factor
        self.G = model.graph
        self.n_s = len(model.states)
        self.n_a = len(model.actions)
        self.model = model

    def get_optimal_policy(self, rewards: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates policy and V of states
        :param rewards: array of mappings state->rewards
        :return: Tuple of policy and Vs. A policy is a mapping where policy[state]->action where state index is given state
        0 - (not look, not talk)
        1 - (not look, talk)
        2 - (look, not talk)
        3 - (look, talk)
        these indices corresponds to model.actions[index]
        """
        return self.__calculate_value_iteration(rewards)

    def __calculate_value_iteration(self, rewards: np.ndarray):

        V = np.zeros(self.n_s)
        policy = np.zeros((self.n_s,))
        i = 0
        while True:
            delta = 0

            for s in self.model.states:

                action_values = np.zeros(self.n_a)
                for a in self.model.actions:
                    index_of_action = self.model.actions.index(a)
                    for prob, next_state in self.G[s][a]:
                        # the reward is the reward of given next_state
                        index_of_next_state = self.model.states.index(next_state)
                        reward = rewards[index_of_next_state]
                        value = prob * (reward + self.discount_factor * V[index_of_next_state])
                        action_values[index_of_action] += value
                        debug = 5
                best_a = np.max(action_values)

                index_of_state = self.model.states.index(s)
                delta = max(delta, np.abs(best_a - V[index_of_state]))
                V[index_of_state] = best_a
            i+=1
            if delta < self.theta:
                break

        for s in self.model.states:
            action_values = np.zeros(self.n_a)
            for a in self.model.actions:
                index_of_action = self.model.actions.index(a)
                for prob, next_state in self.G[s][a]:
                    # the reward is the reward of given next_state
                    index_of_next_state = self.model.states.index(next_state)
                    reward = rewards[index_of_next_state]
                    action_values[index_of_action] += prob * (
                        reward + self.discount_factor * V[index_of_next_state]
                    )
            best_a_index = np.argmax(action_values)
            index_of_state = self.model.states.index(s)
            policy[index_of_state] = best_a_index

        return policy, V
