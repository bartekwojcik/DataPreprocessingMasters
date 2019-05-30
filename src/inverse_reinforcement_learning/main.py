import numpy as np
import settings
import os
import json

from Mdp.mdp_utils import MdpUtils
from Mdp.transition_counting_translator import TransitionCountingTranslator
from inverse_reinforcement_learning.compare_processor import CompareProcessor
from inverse_reinforcement_learning.irl_processor import IrlProcessor
from data_const import UsableConversationConstants as const_uc

if __name__ == "__main__":
    HUMAN_READABLE_FOLDER_PATH = settings.HUMAN_READABLE_FOLDER_PATH
    METADATA_PATH = settings.READABLE_METADATA_FILE_PATH

    with open(METADATA_PATH, "r") as metadata_file:

        metadata_json = json.loads(metadata_file.read())

        for filename in os.listdir(HUMAN_READABLE_FOLDER_PATH):
            full_file_name = os.path.join(HUMAN_READABLE_FOLDER_PATH, filename)

            with open(full_file_name, "r") as conversation_file:

                this_file_metadata = metadata_json[filename]
                conv_json = json.loads(conversation_file.read())

                mdp_graph = MdpUtils.get_at_high_mdp_model()

                VERBOSE = False
                processor = IrlProcessor()
                irl_result = processor.process(
                    conv_json, mdp_graph, this_file_metadata, full_file_name, verbose=VERBOSE
                )

                compare_processor = CompareProcessor()
                compare_processor.compare(
                    irl_result,
                    full_file_name,
                    conv_json,
                    this_file_metadata,
                    settings.TRANSITION_FRAME_STEP,
                    mdp_graph.Ca.shape,
                    show_plot=VERBOSE
                )

                print(f"!!!!!!!!!!!!!!!!!!!!file {filename} done#################################")
                # TODO might do something with irl_result later ¯\_(ツ)_/¯ asd

        debug = 5
