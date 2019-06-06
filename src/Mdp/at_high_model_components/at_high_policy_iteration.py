from Mdp.at_high_model_components.at_high_model import AtHighMdpModel
import numpy as np
from typing import Tuple
import settings


class AtHighPolicyIteration:
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
        return self.__calculate_policy_iteration(rewards)

    def __policy_eval(self,policy,rewards):
        """
        Evaluate a policy given an environment and a full description of the environment's dynamics.
        :param policy:
        :param rewards:
        :return:
        """

        V = np.zeros(self.n_s)
        while True:
            delta = 0
            for s in self.model.states:
                v = 0
                state_index = self.model.states.index(s)
                for action_index, action_proba in enumerate(policy[state_index]):
                    a = self.model.actions[action_index]
                    for prob, next_state in self.G[s][a]:
                        index_of_next_state = self.model.states.index(next_state)
                        reward = rewards[index_of_next_state]
                        v += action_proba * prob * (reward + self.discount_factor * V[index_of_next_state])
                delta = max(delta, abs(v - V[state_index]))
                V[state_index] = v
            if delta < self.theta:
                break

        return np.array(V)



    def __calculate_policy_iteration(self, rewards: np.ndarray):

        policy = np.ones((self.n_s,self.n_a)) / self.n_a

        while True:

            evaluated_policy = self.__policy_eval(policy, rewards)
            policy_stable = True

            for s in self.model.states:
                state_index = self.model.states.index(s)
                current_action_index = np.argmax(policy[state_index])

                actions_values = np.zeros(self.n_a)
                for a in self.model.actions:
                    action_index = self.model.actions.index(a)
                    for prob, next_state in self.G[s][a]:
                        index_of_next_state = self.model.states.index(next_state)
                        reward = rewards[index_of_next_state]
                        actions_values[action_index] += prob * (reward + self.discount_factor * evaluated_policy[index_of_next_state])

                best_action_index = np.argmax(actions_values)
                if best_action_index != current_action_index:
                    policy_stable = False

                policy[state_index] = np.eye(self.n_a)[best_action_index]

            if policy_stable:
                break

        policy_with_numbers_of_actions = np.zeros((self.n_s,))
        for s in self.model.states:
            state_index = self.model.states.index(s)
            policy_with_numbers_of_actions[state_index] = np.argmax(policy[state_index])

        return policy_with_numbers_of_actions, evaluated_policy


