import settings
import os
import json

from inverse_reinforcement_learning.mdp_utils import MdpUtils
from inverse_reinforcement_learning.policy_parser import PolicyParser
from inverse_reinforcement_learning.policy_trans_matrix_creator import PolicyMatrixCreator

if __name__ == '__main__':


    FILE_TO_READ = os.path.join(settings.HUMAN_READABLE_FOLDER_PATH, "human_conversation_4.json")

    with open(FILE_TO_READ, "r") as conversation_file:

        conv_json = json.loads(conversation_file.read())
        optimal_policy = PolicyParser.parse_data_to_policy(conv_json)
        mdp_graph = MdpUtils.simple_16_action_graph()
        actions_names = mdp_graph.get_actions_names()
        states_names = mdp_graph.get_states_names()

        matrix_creator = PolicyMatrixCreator(optimal_policy,actions_names,states_names)





        debug = 5