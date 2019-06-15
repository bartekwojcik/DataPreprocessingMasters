from collections import defaultdict

import numpy as np

###
# some code was used from https://github.com/dennybritz/reinforcement-learning/blob/master/TD/Q-Learning%20Solution.ipynb
from Mdp.at_high_model_components import environment
from Mdp.at_high_model_components.environment import Environment


class QLearner:

    def __init__(self,
                 q_iterations: int,
                 discount_factor:float,
                 q_alpha:float,
                 episode_length,
                 policy_epsilon= 0.05):

        self.episode_length = episode_length
        self.discount_factor = discount_factor
        self.q_alpha = q_alpha
        self.policy_epsilon = policy_epsilon
        self.q_iterations = q_iterations

    def __make_epsilon_greedy_policy(self,Q, epsilon, nA):
        def policy_fn(observation):
            A = np.ones(nA, dtype=float) * epsilon / nA
            best_action = np.argmax(Q[observation])
            A[best_action] += (1.0 - epsilon)
            return A

        return policy_fn

    def learn(self, env:Environment):
        """
        :return: Q values
        """
        Q = defaultdict(lambda: np.zeros(env.n_actions))
        policy = self.__make_epsilon_greedy_policy(Q, self.policy_epsilon, env.n_actions)

        for i_episode in range(self.q_iterations):
            state = env.reset()

            for t in range(self.episode_length):
                action_probas = policy(state)
                # i keep forgeting that we "have to" use epsilon greedy policy
                # action = np.argmax(action_probas)
                action_index = np.random.choice(np.arange(len(action_probas)), p=action_probas)
                action = env.model.actions[action_index]
                new_state, reward = env.step(action)

                next_action_probas = Q[new_state]
                a_max = np.argmax(next_action_probas)
                q_prim = Q[new_state][a_max]

                Q[state][action_index] += self.q_alpha * (reward + self.discount_factor * q_prim - Q[state][action_index])
                state = new_state

        return Q