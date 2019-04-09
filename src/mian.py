import os
import json
import settings

from conversation_gaze_translator import ConversationGazeTranslator

if __name__ == "__main__":

    JOINT_FILE_PATH = os.path.join(
        settings.JOINT_DATA_FOLDER_PATH, "conversation_01.json"
    )
    CLUSTER_FILE_PATH = os.path.join(
        settings.CLUSTER_DATA_FOLDER_PATH, "clustering_01.json"
    )

    with open(JOINT_FILE_PATH, "r") as joint_file, open(
        CLUSTER_FILE_PATH, "r"
    ) as cluster_file:

        joint_data_string = joint_file.read()
        cluster_data_string = cluster_file.read()

        joint_data_parsed = json.loads(joint_data_string)
        cluster_data_parsed = json.loads(cluster_data_string)

        conversation_translator = ConversationGazeTranslator(joint_data_parsed, cluster_data_parsed)

