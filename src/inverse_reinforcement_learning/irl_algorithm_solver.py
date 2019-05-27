from typing import List, Tuple
import pprint


from Mdp.at_high_model_components.at_high_model_value_iteration import (
    AtHighValueIteration,
)
from Mdp.at_high_model_components.at_high_policy_player import HighPolicyPlayer

from inverse_reinforcement_learning.feature_expectations_extractor import (
    FeatureExpectationExtractor,
)
from inverse_reinforcement_learning.reward_calculator import RewardCalculator
import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt


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
        value_iterator: AtHighValueIteration,
        feature_expectation_extractor: FeatureExpectationExtractor,
        policy_player: HighPolicyPlayer,
        epsilon=0.1,
        policy_player_max_step:int = 5000,
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
        plt.axis([0, 50, 0, 10])
        pp = pprint.PrettyPrinter(indent=4)
        i = 0
        while True:

            try:
                W = self.calc_weights()
            except ValueError:
                return W, reward_matrix, policy, V, new_conversation, False
            self.current_t, reward_matrix, policy, V, new_conversation = self.update_policy_list(W)
            print(f"iteration: {i}")
            print("weights:")
            pp.pprint(W)
            print("##### reward matrix ########")
            pp.pprint(reward_matrix)
            print("###### policy ######")
            pp.pprint(policy)

            if verbose:
                plt.scatter(i, self.current_t)
                plt.pause(0.05)

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

    def update_policy_list(self, W: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, List[dict] ]:
        """

        :param self:
        :param W:
        :return: hyper_distance, reward_matrix, policy, V, new_conversation
        """
        # get feature expectations of a new policy respective to the input weights
        temp_fe, reward_matrix, policy, V, new_conversation = self.get_reinforcement_learning_features_expectations(W)
        hyper_distance = np.abs(
            np.dot(
                W, np.asarray(self.expert_feature_expectations) - np.asarray(temp_fe)
            )
        )  # hyperdistance = t
        self.policies_feature_expectations[hyper_distance] = temp_fe
        # t = (weights.transpose)*(expert-newPolicy)
        return (hyper_distance, reward_matrix, policy, V, new_conversation)

    def get_reinforcement_learning_features_expectations(
        self, W: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """

        :param W: weights
        :return: new_features, reward_matrix, policy, V, new_conversation
        """
        if np.any(np.isnan(W)):
            raise ValueError("some elements of W are Nan")

        reward_matrix = self.reward_calculator.calculate_reward(W)
        policy, V = self.value_iterator.get_optimal_policy(reward_matrix)
        new_conversation = self.policy_player.play_policy(policy, max_steps= self.policy_player_max_step)
        new_features = self.feature_expectation_extractor.get_experts_feature_expectations(
            new_conversation
        )

        return new_features, reward_matrix, policy, V, new_conversation
