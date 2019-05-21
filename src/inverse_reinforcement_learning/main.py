
import numpy as np
import settings
import os
import json

from Mdp.mdp_utils import MdpUtils
from Mdp.transition_counting_translator import TransitionCountingTranslator
from inverse_reinforcement_learning.compare_processor import CompareProcessor
from inverse_reinforcement_learning.irl_processor import IrlProcessor

if __name__ == "__main__":

    FILE_NAME = "human_readable_conversation_9.json"
    FILE_TO_READ = os.path.join(settings.HUMAN_READABLE_FOLDER_PATH, FILE_NAME)
    METADATA_PATH = settings.READABLE_METADATA_FILE_PATH

    with open(METADATA_PATH, "r") as metadata_file:
        metadata_json = json.loads(metadata_file.read())

        transition_counting_array = np.load(
            os.path.join(
                settings.MY_DATA_FOLDER_PATH, "transition_counting_results.npy"
            )
        )
        translator = TransitionCountingTranslator(transition_counting_array)

        with open(FILE_TO_READ, "r") as conversation_file:

            this_file_metadata = metadata_json[FILE_NAME]
            conv_json = json.loads(conversation_file.read())

            mdp_graph = MdpUtils.get_at_high_mdp_model()

            processor = IrlProcessor()
            irl_result = processor.process(conv_json,mdp_graph,this_file_metadata,FILE_NAME)

            compare_processor = CompareProcessor()
            compare_processor.compare(irl_result,FILE_NAME,conv_json,this_file_metadata,12,True)


        debug = 5
