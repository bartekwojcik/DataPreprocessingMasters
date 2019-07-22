from typing import List

import numpy as np

from Mdp.at_high_model_components.at_high_model import AtHighMdpModel
from Mdp.at_high_model_components.at_high_policy_player import HighPolicyPlayer
from Mdp.at_high_model_components.q_learning import QLearner
from inverse_reinforcement_learning.feature_expectations_extractor import (
    FeatureExpectationExtractor,
)
from inverse_reinforcement_learning.irl_algorithm_solver import IrlAlgorithmSolver
from inverse_reinforcement_learning.irl_processor_result import IrlProcessorResult
from inverse_reinforcement_learning.reward_calculator import RewardCalculator
from settings import Settings


class IrlProcessor:
    """
    Encapsulates IRL algorithm process, it is just a facade
    """

    def process(
        self,
        conversation_json: List[dict],
        mdp_graph: AtHighMdpModel,
        metadata: dict,
        file_name: str,
        policy_player: HighPolicyPlayer,
        policy_player_max_step: int,
        verbose: bool,
        irl_solver_iterations: int,
        q_learning_episode_length: int,
            Q_ITERATIONS,
            DISCOUNT_FACTOR,
            Q_ALPHA,
            Q_EPSILON,
            IRL_SOLVER_EPSILON
    ) -> IrlProcessorResult:
        """
        Processes one file of conversation with Inverse Reinforcement Learning
        Paramters are just like the ones in inverse inverse_reinforcement_learning.process_file
        """

        feature_expectation_extractor = FeatureExpectationExtractor(
            mdp_graph.states, metadata, 0.9999999
        )

        expert_feature_expectations = feature_expectation_extractor.get_feature_expectations(
            conversation_json
        )

        random_feature_expectations = feature_expectation_extractor.get_random_feature_expectations(
            len(conversation_json), mdp_graph, policy_player
        )

        states_array = np.array(mdp_graph.states)
        reward_calculator = RewardCalculator(states_array.shape, mdp_graph.states)
        # value_iterator = AtHighValueIteration(mdp_graph,settings.POLICY_THETA,settings.DISCOUNT_FACTOR)

        q_learner = QLearner(
            Q_ITERATIONS,
            DISCOUNT_FACTOR,
            Q_ALPHA,
            q_learning_episode_length,
            Q_EPSILON,
        )

        irl = IrlAlgorithmSolver(
            file_name,
            expert_feature_expectations,
            random_feature_expectations,
            reward_calculator,
            feature_expectation_extractor,
            policy_player,
            q_learner,
            mdp_graph,
            policy_player_max_step=policy_player_max_step,
            epsilon=IRL_SOLVER_EPSILON,
            max_iterations=irl_solver_iterations,
        )

        weights, reward_matrix, policy, Q, new_conversation, is_ok, list_of_t_W_intercept = irl.find_weights(
            verbose=verbose
        )

        return IrlProcessorResult(
            weights,
            reward_matrix,
            policy,
            Q,
            new_conversation,
            is_ok,
            list_of_t_W_intercept,
        )
