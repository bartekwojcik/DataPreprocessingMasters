import os
import json
import settings

from conversation_gaze_translator import ConversationGazeTranslator

if __name__ == "__main__":
    JOINT_FILE_PATH = os.path.join(
        settings.JOINT_DATA_FOLDER_PATH, "conversation_04.json"
    )
    CLUSTER_FILE_PATH = os.path.join(
        settings.CLUSTER_DATA_FOLDER_PATH, "clustering_04.json"
    )

    with open(JOINT_FILE_PATH, "r") as joint_file, open(
        CLUSTER_FILE_PATH, "r"
    ) as cluster_file, open(
        settings.USABLE_CONVERSATIONS_FILE_PATH, "r"
    ) as usable_conversation_file:
        joint_data_string = joint_file.read()
        cluster_data_string = cluster_file.read()

        joint_data_parsed = json.loads(joint_data_string)
        cluster_data_parsed = json.loads(cluster_data_string)

        # this should be per usable_conversation, extracting given file and then opening files
        conversation_translator = ConversationGazeTranslator(
            joint_data_parsed, cluster_data_parsed, "person1", "person2", 3
        )
        labeled_data = conversation_translator.convert_to_readable()
