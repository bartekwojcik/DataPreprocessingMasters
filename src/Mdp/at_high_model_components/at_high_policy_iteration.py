from Mdp.at_high_model_components.at_high_model import AtHighMdpModel
import numpy as np
from typing import Tuple


class AtHighPolicyIteration:
    def __init__(
        self, model: AtHighMdpModel, theta=0.0001, discount_factor=0.95
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
                    list_of_s_prime = self.G[s][a]
                    for prob, next_state in list_of_s_prime:
                        index_of_next_state = self.model.states.index(next_state)
                        reward = rewards[index_of_next_state]
                        v += action_proba * prob * (reward + self.discount_factor * V[next_state])
                delta = max(delta, abs(v - V[state_index]))
                V[state_index] = v
            if delta < self.theta:
                break
        return np.array(V)

    def __calculate_policy_iteration(self, rewards: np.ndarray):

        policy = np.zeros((self.n_s,))

        while True:

            evaluated_policy = self.__policy_eval(policy, rewards)
            policy_stable = True

            for s in range(env.nS):
                current_action = np.argmax(policy[s])

                actions_values = np.zeros(env.nA)
                for a in range(env.nA):
                    for prob, next_state, reward, done in env.P[s][a]:
                        actions_values[a] += prob * (reward + discount_factor * evaluated_policy[next_state])

                best_action = np.argmax(actions_values)
                if best_action != current_action:
                    policy_stable = False

                policy[s] = np.eye(env.nA)[best_action]

            if policy_stable:
                break

        return policy, evaluated_policy


