import settings
import os
import json

from Mdp.mdp_utils import MdpUtils
from Mdp.policy_parser import PolicyParser

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






        debug = 5