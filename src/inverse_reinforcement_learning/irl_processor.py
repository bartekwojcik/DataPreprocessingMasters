import random
from typing import List, Tuple
import numpy as np
from Mdp.at_high_model_components.at_high_model import AtHighMdpModel
from Mdp.at_high_model_components.at_high_model_value_iteration import AtHighValueIteration
from Mdp.at_high_model_components.at_high_policy_player import HighPolicyPlayer
from Mdp.mdp_utils import MdpUtils
from inverse_reinforcement_learning.feature_expectations_extractor import FeatureExpectationExtractor
from inverse_reinforcement_learning.irl_algorithm_solver import IrlAlgorithmSolver
from inverse_reinforcement_learning.irl_processor_result import IrlProcessorResult
from inverse_reinforcement_learning.reward_calculator import RewardCalculator


class IrlProcessor():

    def __get_random_feature_expectations(self) -> np.ndarray:
        n_states = len(MdpUtils.get_at_high_mdp_model().states)
        random_list = [random.uniform(0, 5) for i in range(n_states)]
        return np.array(random_list)

    def process(self, conversation_json: List[dict], mdp_graph: AtHighMdpModel, metadata: dict, file_name: str)->IrlProcessorResult:
        """
        Processes one file of conversation with Inverse Reinforcement Learning
        :param conversation_json:
        :param mdp_graph:
        :param metadata:
        :return: IrlProcessorResult
        """
        feature_expectation_extractor = FeatureExpectationExtractor(
            len(mdp_graph.states), metadata
        )

        expert_feature_expectations = feature_expectation_extractor.get_experts_feature_expectations(
            conversation_json
        )
        random_feature_expectations = self.__get_random_feature_expectations()
        policy_player = HighPolicyPlayer(metadata, mdp_graph)

        reward_calculator = RewardCalculator(len(mdp_graph.states))
        value_iterator = AtHighValueIteration(mdp_graph)
        irl = IrlAlgorithmSolver(
            file_name,
            expert_feature_expectations,
            random_feature_expectations,
            reward_calculator,
            value_iterator,
            feature_expectation_extractor,
            policy_player,
            policy_player_max_step=2000,
            max_iterations=50

        )
        weights, reward_matrix, policy, V, new_conversation, is_ok = irl.find_weights(
            verbose=True
        )

        return IrlProcessorResult(weights, reward_matrix, policy, V, new_conversation, is_ok)

