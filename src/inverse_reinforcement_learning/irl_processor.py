from typing import List

import numpy as np

from Mdp.at_high_model_components.environment import Environment
from Mdp.at_high_model_components.q_learning import QLearner
from settings import Settings
from Mdp.at_high_model_components.at_high_model import AtHighMdpModel
from Mdp.at_high_model_components.at_high_model_value_iteration import (
    AtHighValueIteration,
)
from Mdp.at_high_model_components.at_high_policy_player import HighPolicyPlayer
from inverse_reinforcement_learning.feature_expectations_extractor import (
    FeatureExpectationExtractor,
)
from inverse_reinforcement_learning.irl_algorithm_solver import IrlAlgorithmSolver
from inverse_reinforcement_learning.irl_processor_result import IrlProcessorResult
from inverse_reinforcement_learning.reward_calculator import RewardCalculator


class IrlProcessor:
    """
    Encapsulates IRL algorithm process
    """

    def process(
        self,
        conversation_json: List[dict],
        mdp_graph: AtHighMdpModel,
        metadata: dict,
        file_name: str,
        policy_player_max_step,
        verbose: bool,
        settings:Settings,
        irl_solver_iterations:int
    ) -> IrlProcessorResult:
        """
        Processes one file of conversation with Inverse Reinforcement Learning
        :param conversation_json:
        :param mdp_graph:
        :param metadata:
        :return: IrlProcessorResult
        """

        policy_player = HighPolicyPlayer(metadata, mdp_graph,0.05)


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
        #value_iterator = AtHighValueIteration(mdp_graph,settings.POLICY_THETA,settings.DISCOUNT_FACTOR)

        q_learner = QLearner(settings.Q_ITERATIONS, settings.DISCOUNT_FACTOR, settings.Q_ALPHA)

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
            epsilon= settings.IRL_SOLVER_EPSILON,
            max_iterations=irl_solver_iterations
        )
        #TODO change this to include these Q values
        #TODO add rewards tracking later so you can plot 3D things

        weights, reward_matrix, policy, new_conversation, is_ok, list_of_t_W_intercept = irl.find_weights(
            verbose=verbose
        )

        #new_conversation_with_epsilon = HighPolicyPlayer(metadata,mdp_graph,0.05).play_policy(policy,policy_player_max_step)

        return IrlProcessorResult(
            weights, reward_matrix, policy, V, new_conversation, is_ok, list_of_t_W_intercept
        )
