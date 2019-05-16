from typing import List

import numpy as np
from cvxopt import matrix, solvers


class IrlAlgorithmSolver:
    """
    Solves Inverse Reinforcement Learning problem

    Uses data found at:

    http://3dvision.princeton.edu/courses/COS598/2014sp/slides/lecture07_reinforcement.pdf

    http://ai.stanford.edu/~ang/papers/icml04-apprentice.pdf

    https://jangirrishabh.github.io/2016/07/09/virtual-car-IRL/
    """
    def __init__(self, expert_feature_expectations,random_feature_expectations, epsilon=0.1):

        self.random_feature_expectations = random_feature_expectations
        self.epsilon = epsilon
        self.expert_feature_expectations = expert_feature_expectations

        # step 1 of algorithm
        # norm of the diff in expert and random
        self.random_t = np.linalg.norm(np.asarray(self.expert_feature_expectations) - np.asarray(self.random_feature_expectations))

        # storing the policies and their respective t values in a dictionary
        self.policies_feature_expectations = {self.random_t: self.random_feature_expectations}

        self.currentT = self.random_t

    def find_weights(self) -> np.ndarray:

        while True:
            W = self.calc_weights()
            self.current_t = self.update_policy_list(W)
            if self.current_t <= self.epsilon:
                #step 3
                break

        assert W, "weights are broken"
        return W

    def calc_weights(self)-> np.ndarray:
        """
        Step 2 of the algorithm
        taken from https://jangirrishabh.github.io/2016/07/09/virtual-car-IRL/
        finds the maximum margin hyperplane separating two sets of points.
        :return: weights
        """
        m = len(self.expert_feature_expectations)  # feature expectation
        P = matrix(2.0 * np.eye(m), tc='d')  # min ||w||
        q = matrix(np.zeros(m), tc='d')
        policy_list = [self.expert_feature_expectations]
        h_list = [1]
        for i in self.policies_feature_expectations.keys():
            policy_list.append(self.policies_feature_expectations[i])
            h_list.append(1)
        policy_mat = np.matrix(policy_list)
        policy_mat[0] = -1 * policy_mat[0]
        G = matrix(policy_mat, tc='d')
        h = matrix(-np.array(h_list), tc='d')
        sol = solvers.qp(P, q, G, h)

        weights = np.squeeze(np.asarray(sol['x']))
        norm = np.linalg.norm(weights)
        weights = weights / norm
        return weights  # return the normalized weights

    def update_policy_list(self, W: np.ndarray)->int:
        # get feature expectations of a new policy respective to the input weights
        temp_fe = self.get_reinforcement_learning_features_expectations(W)
        hyper_distance = np.abs(np.dot(W, np.asarray(self.expert_feature_expectations) - np.asarray(temp_fe)))  # hyperdistance = t
        self.policies_feature_expectations[hyper_distance] = temp_fe
        # t = (weights.tanspose)*(expert-newPolicy)
        return hyper_distance

    def get_reinforcement_learning_features_expectations(self, W)-> List[int]:
        # TODO Step 4 of the algorithm
        # using value W you have to make it run to produce feature expectations.
        # TODO how to use W in 'playing out' agent and model?

        pass

