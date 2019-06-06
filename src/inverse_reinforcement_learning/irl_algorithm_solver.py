from typing import List, Tuple, Union

import settings
from Mdp.at_high_model_components.at_high_model_value_iteration import (
    AtHighValueIteration,
)
from Mdp.at_high_model_components.at_high_policy_iteration import AtHighPolicyIteration
from Mdp.at_high_model_components.at_high_policy_player import HighPolicyPlayer

from inverse_reinforcement_learning.feature_expectations_extractor import (
    FeatureExpectationExtractor,
)
from inverse_reinforcement_learning.reward_calculator import RewardCalculator
import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
from sklearn.svm import SVC


class IrlAlgorithmSolver:
    """
    Solves Inverse Reinforcement Learning problem

    Uses data found at:

    http://3dvision.princeton.edu/courses/COS598/2014sp/slides/lecture07_reinforcement.pdf

    http://ai.stanford.edu/~ang/papers/icml04-apprentice.pdf

    https://jangirrishabh.github.io/2016/07/09/virtual-car-IRL/
    """

    def __init__(
        self,
        conversation_name: str,
        expert_feature_expectations: np.ndarray,
        random_feature_expectations: np.ndarray,
        reward_calculator: RewardCalculator,
        value_iterator: Union[AtHighValueIteration,AtHighPolicyIteration],
        feature_expectation_extractor: FeatureExpectationExtractor,
        policy_player: HighPolicyPlayer,
        policy_player_max_step:int,
        epsilon= settings.IRL_SOLVER_EPSILON,
        max_iterations = 100
    ):
        """
        :param policy_player: policy player
        :param feature_expectation_extractor: calculates feature expectations
        :param value_iterator: objects that calculates value iteration algorithm
        :param expert_feature_expectations:
        :param random_feature_expectations:
        :param reward_calculator:
        :param epsilon:
        """
        self.conversation_name = conversation_name
        self.max_iterations = max_iterations
        self.policy_player_max_step = policy_player_max_step
        self.policy_player = policy_player
        self.feature_expectation_extractor = feature_expectation_extractor
        self.value_iterator = value_iterator
        self.reward_calculator = reward_calculator
        self.random_feature_expectations = random_feature_expectations
        self.epsilon = epsilon
        self.expert_feature_expectations = expert_feature_expectations

        #just for tests
        self.previous_W = 0
        self.previous_intercept = 0
        self.previous_policy = np.zeros((1,1))

        # step 1 of algorithm
        # norm of the diff in expert and random
        self.random_t = np.linalg.norm(
            np.asarray(self.expert_feature_expectations)
            - np.asarray(self.random_feature_expectations)
        )

        # storing the policies and their respective t values in a dictionary
        self.policies_feature_expectations = {
            self.random_t: self.random_feature_expectations
        }

        self.currentT = self.random_t

    def find_weights(self,verbose: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[dict], bool ]:
        """

        :return: weights, reward_matrix, policy, V, new_conversation, is_ok_or_broken_after_max_iters (True =ok , False = fucked_up)?

        """
        #plt.axis([0, 50, 0, 10])

        i = 0
        while True:

            # try:
            #     W = self.calc_weights()
            # except ValueError:
            #     print(f"{self.conversation_name} Value iteration happened#################################")
            #     return W, reward_matrix, policy, V, new_conversation, False

            W, intercept = self.get_SVM_weights()

            self.current_t, reward_matrix, policy, V, new_conversation = self.update_policy_list(W, intercept)

            if verbose:
                plt.scatter(i, self.current_t)
                plt.pause(0.05)
            print(f"iteration:{i}, current t: {self.current_t}")
            if self.current_t <= self.epsilon:
                # step 3
                break
            if i > self.max_iterations:
                print(f"{self.conversation_name} file IS STUCK AND ITERATION IS BROKEN")
                return W, reward_matrix, policy, V, new_conversation, False
            i += 1

        assert not (np.all(np.isnan(W))), "weights are broken"

        return W, reward_matrix, policy, V, new_conversation, True

    def calc_weights(self) -> np.ndarray:
        """
        Step 2 of the algorithm
        taken from https://jangirrishabh.github.io/2016/07/09/virtual-car-IRL/
        finds the maximum margin hyperplane separating two sets of points.
        :return: weights
        """
        m = len(self.expert_feature_expectations)  # feature expectation
        P = matrix(2.0 * np.eye(m), tc="d")  # min ||w||
        q = matrix(np.zeros(m), tc="d")
        policy_list = [self.expert_feature_expectations]
        h_list = [1]
        for i in self.policies_feature_expectations.keys():
            #get just t
            policy_list.append(self.policies_feature_expectations[i])
            h_list.append(1)
        policy_mat = np.matrix(policy_list)
        policy_mat[0] = -1 * policy_mat[0]
        G = matrix(policy_mat, tc="d")
        h = matrix(-np.array(h_list), tc="d")
        sol = solvers.qp(P, q, G, h)

        weights = np.squeeze(np.asarray(sol["x"]))
        norm = np.linalg.norm(weights)
        weights = weights / norm
        return weights  # return the normalized weights

    def get_SVM_weights(self):

        shape = (len(self.policies_feature_expectations)+1,len(self.expert_feature_expectations))
        X = np.zeros(shape)
        Y = np.ones((shape[0],))
        clf = SVC(kernel='linear')

        for i, (t, feature_expectation) in enumerate(self.policies_feature_expectations.items()):
            X[i+1] = feature_expectation
            Y[i+1] = -1

        clf.fit(X,Y)

        W = np.squeeze(clf.coef_)

        return W, clf.intercept_


    def update_policy_list(self, W: np.ndarray, intercept:int) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, List[dict] ]:
        """

        :param self:
        :param W:
        :return: hyper_distance, reward_matrix, policy, V, new_conversation
        """

        # get feature expectations of a new policy respective to the input weights
        temp_fe, reward_matrix, policy, V, new_conversation = self.get_reinforcement_learning_features_expectations(W, intercept)
        hyper_distance = np.abs(
            np.dot(
                W, np.asarray(self.expert_feature_expectations) - np.asarray(temp_fe)
            ) + intercept
        )

        # hyperdistance = t
        float_hyper_distance = float(hyper_distance)
        self.policies_feature_expectations[float_hyper_distance] = temp_fe

        # t = (weights.transpose)*(expert-newPolicy)
        return (float_hyper_distance, reward_matrix, policy, V, new_conversation)

    def get_reinforcement_learning_features_expectations(
        self, W: np.ndarray, intercept:int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """

        :param W: weights
        :return: new_features, reward_matrix, policy, V, new_conversation
        """
        if np.any(np.isnan(W)):
            raise ValueError("some elements of W are Nan")



        reward_matrix = self.reward_calculator.calculate_reward(W, intercept)
        policy, V = self.value_iterator.get_optimal_policy(reward_matrix)
        new_conversation = self.policy_player.play_policy(policy, max_steps= self.policy_player_max_step)
        new_features = self.feature_expectation_extractor.get_feature_expectations(
            new_conversation
        )

        print(f"sum of new features:{np.sum(new_features)}")
        print(f"sum of policy:{np.sum(policy)}")
        print(f"sum of rewards:{np.sum(reward_matrix)}")
        print(f"sum of W:{np.sum(W)}")
        print(f"intercept: {intercept}")

        print(f"sum of W difference:{'{:.10f}'.format(np.sum(W) - np.sum(self.previous_W))}")
        print(f"intercept difference:{'{:.10f}'.format(np.sum(intercept) - np.sum(self.previous_intercept))}")


        self.previous_reward_matrix = np.array(reward_matrix)
        self.previous_W = np.array(W)
        self.previous_policy = np.array(policy)
        self.previous_intercept = intercept
        return new_features, reward_matrix, policy, V, new_conversation
