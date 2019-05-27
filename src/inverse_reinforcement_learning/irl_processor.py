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
    """
    Encapsulates IRL algorithm process
    """
    def __get_random_feature_expectations(self, n_attributes) -> np.ndarray:
        """

        :param n_states: number of attributes in state
        :return:
        """

        random_list = [random.uniform(0, 5) for i in range(n_attributes)]
        return np.array(random_list)

    def process(self, conversation_json: List[dict], mdp_graph: AtHighMdpModel, metadata: dict, file_name: str,verbose:bool)->IrlProcessorResult:
        """
        Processes one file of conversation with Inverse Reinforcement Learning
        :param conversation_json:
        :param mdp_graph:
        :param metadata:
        :return: IrlProcessorResult
        """
        n_attributes_in_state =len(mdp_graph.states[0])
        feature_expectation_extractor = FeatureExpectationExtractor(
            n_attributes_in_state, metadata
        )

        expert_feature_expectations = feature_expectation_extractor.get_experts_feature_expectations(
            conversation_json
        )

        random_feature_expectations = self.__get_random_feature_expectations(n_attributes_in_state)
        policy_player = HighPolicyPlayer(metadata, mdp_graph)

        states_array = np.array(mdp_graph.states)
        reward_calculator = RewardCalculator(states_array.shape, states_array)
        value_iterator = AtHighValueIteration(mdp_graph)
        irl = IrlAlgorithmSolver(
            file_name,
            expert_feature_expectations,
            random_feature_expectations,
            reward_calculator,
            value_iterator,
            feature_expectation_extractor,
            policy_player,
            policy_player_max_step=3000,
            max_iterations=50

        )
        weights, reward_matrix, policy, V, new_conversation, is_ok = irl.find_weights(
            verbose=verbose
        )

        return IrlProcessorResult(weights, reward_matrix, policy, V, new_conversation, is_ok)

