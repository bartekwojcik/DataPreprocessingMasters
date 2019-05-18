import random

import numpy as np
import settings
import os
import json

from Mdp.policy_players.simple_deterministic_policy_player import SimpleDeterministicPolicyPlayer
from Mdp.dynamic_programming_algorithms.simple_model_value_iteration import SimpleModelValueIteration
from Mdp.mdp_utils import MdpUtils
from inverse_reinforcement_learning.irl_algorithm_solver import IrlAlgorithmSolver
from inverse_reinforcement_learning.feature_expectations_extractor import (
    FeatureExpectationExtractor,
)
from inverse_reinforcement_learning.reward_calculator import RewardCalculator


def get_random_feature_expectations() -> np.ndarray:
    # TODO this is mock
    # for now, features mapping might be only [high_state, low_state]
    # TODO TO DISCUSS
    random_list = [random.randint(2, 8) for n in range(4)]
    return np.array(random_list)


if __name__ == "__main__":

    FILE_NAME = "human_readable_conversation_4.json"
    FILE_TO_READ = os.path.join(settings.HUMAN_READABLE_FOLDER_PATH, FILE_NAME)
    METADATA_PATH = settings.READABLE_METADATA_FILE_PATH

    with open(METADATA_PATH, "r") as metadata_file:
        metadata_json = json.loads(metadata_file.read())

        with open(FILE_TO_READ, "r") as conversation_file:

            this_file_metadata = metadata_json[FILE_NAME]
            conv_json = json.loads(conversation_file.read())
            # optimal_policy = PolicyParser.parse_data_to_policy(conv_json,this_file_metadata)
            mdp_graph = MdpUtils.simple_16_action_graph()

            feature_expectation_extractor = FeatureExpectationExtractor(
                len(mdp_graph.states), this_file_metadata
            )

            expert_feature_expectations = feature_expectation_extractor.get_experts_feature_expectations(conv_json)
            random_feature_expectations = get_random_feature_expectations()
            policy_player = SimpleDeterministicPolicyPlayer(this_file_metadata)

            reward_calculator = RewardCalculator(len(mdp_graph.states))
            value_iterator = SimpleModelValueIteration(mdp_graph)
            irl = IrlAlgorithmSolver(
                expert_feature_expectations,
                random_feature_expectations,
                reward_calculator,
                value_iterator,
                feature_expectation_extractor,
                policy_player
            )
            weights = irl.find_weights()

            # TODO things to do:
            # 1) DONE implement calculating feature expectations from dataset (easy)
            # 2) implement playing out game with weights (W) to obtain more feature expectations
            # 3) how to include many experts policies (from all the files)
            # 4) what exactly does the algorithm return?

        debug = 5
