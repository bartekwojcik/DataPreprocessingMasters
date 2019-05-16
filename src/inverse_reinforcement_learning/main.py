import random
from typing import List

import settings
import os
import json

from Mdp.mdp_utils import MdpUtils
from Mdp.policy_parser import PolicyParser
from inverse_reinforcement_learning.IrlAlgorithmSolver import IrlAlgorithmSolver


def get_mock_feature_expectations()->List[int]:
    #TODO this is mock
    # for now, features mapping might be only [high_state, low_state]
    # TODO TO DISCUSS
    random_list = [random.randint(2,8) for n in range(4)]
    return random_list




if __name__ == '__main__':

    FILE_NAME = "human_readable_conversation_4.json"
    FILE_TO_READ = os.path.join(settings.HUMAN_READABLE_FOLDER_PATH, FILE_NAME)
    METADATA_PATH = settings.READABLE_METADATA_FILE_PATH

    with open(METADATA_PATH, "r") as metadata_file:
        metadata_json = json.loads(metadata_file.read())

        with open(FILE_TO_READ, "r") as conversation_file:

            this_file_metadata = metadata_json[FILE_NAME]
            conv_json = json.loads(conversation_file.read())
            optimal_policy = PolicyParser.parse_data_to_policy(conv_json,this_file_metadata)
            mdp_graph = MdpUtils.simple_16_action_graph()

            expert_feature_expectations: List[int] = get_mock_feature_expectations()
            random_feature_expectations = get_mock_feature_expectations()
            irl = IrlAlgorithmSolver(random_feature_expectations,random_feature_expectations)
            weights = irl.find_weights()

            #TODO things to do:
            # 1) implement calculating feature expectations from dataset (easy)
            # 2) implement playing out game with weights (W) to obtain more feature expectations
            # 3) how to include many experts policies (from all the files)
            # 4) what exactly does the algorithm return?


        debug = 5

