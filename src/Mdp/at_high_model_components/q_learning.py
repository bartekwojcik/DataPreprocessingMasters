from collections import defaultdict

import numpy as np

###
# some code was used from https://github.com/dennybritz/reinforcement-learning/blob/master/TD/Q-Learning%20Solution.ipynb
from Mdp.at_high_model_components import Environment


class QLearner:

    def __make_epsilon_greedy_policy(self,Q, epsilon, nA):
        def policy_fn(observation):
            A = np.ones(nA, dtype=float) * epsilon / nA
            best_action = np.argmax(Q[observation])
            A[best_action] += (1.0 - epsilon)
            return A

        return policy_fn

    def learn(self, env:Environment, num_episodes, discount_factor, alpha, epsilon):

        Q = defaultdict(lambda: np.zeros(env.n_actions))
        policy = self.__make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

        for i_episode in range(num_episodes):
            state = env.reset()
            done = False
            t = 0
            while not done:
                action_probas = policy(state)
                # i keep forgeting that we "have to" use epsilon greedy policy
                # action = np.argmax(action_probas)
                action = np.random.choice(np.arange(len(action_probas)), p=action_probas)
                new_state, reward = env.step(action)

                next_action_probas = Q[new_state]
                a_max = np.argmax(next_action_probas)
                q_prim = Q[new_state][a_max]

                Q[state][action] += alpha * (reward + discount_factor * q_prim - Q[state][action])
                state = new_state

                t += 1

        return Q