import os
import json
import settings
from data_const import UsableConversationConstants as ConstUC
from conversation_gaze_translator import ConversationGazeTranslator

if __name__ == "__main__":
    JOINT_FILE_PATH = os.path.join(
        settings.JOINT_DATA_FOLDER_PATH, "conversation_04.json"
    )
    CLUSTER_FILE_PATH = os.path.join(
        settings.CLUSTER_DATA_FOLDER_PATH, "clustering_04.json"
    )


    with open(
        settings.USABLE_CONVERSATIONS_FILE_PATH, "r"
    ) as usable_conversation_file:

        usable_conversations_data = json.loads(usable_conversation_file.read())[ConstUC.CONVERSATIONS]

        for conversation_data in usable_conversations_data:

            conv_number = conversation_data[ConstUC.NUMBER]
            joint_file_path = os.path.join(settings.JOINT_DATA_FOLDER_PATH,f"conversation_{str(conv_number).zfill(2)}.json")
            cluster_file_path = os.path.join(settings.CLUSTER_DATA_FOLDER_PATH, f"clustering_{str(conv_number).zfill(2)}.json")
            people_dict = {ConstUC.PERSON1: ConstUC.PERSON2, ConstUC.PERSON2: ConstUC.PERSON1}
            main_person = conversation_data[ConstUC.MAIN]
            other_person = people_dict[main_person]
            main_person_clusters = conversation_data[ConstUC.FACE_CLUSTERS]
            other_person_clusters = conversation_data[ConstUC.OTHER_PERSON_CLUSTERS]


            with open(joint_file_path, "r") as joint_file, open(
                    cluster_file_path, "r"
            ) as cluster_file:
                joint_data_string = joint_file.read()
                cluster_data_string = cluster_file.read()

                joint_data_parsed = json.loads(joint_data_string)
                cluster_data_parsed = json.loads(cluster_data_string)

                # this should be per usable_conversation, extracting given file and then opening files
                conversation_translator = ConversationGazeTranslator(
                    joint_data_parsed, cluster_data_parsed, main_person, other_person, main_person_clusters, other_person_clusters
                )
                labeled_data = conversation_translator.convert_to_readable()

                folder_path = settings.HUMAN_READABLE_FOLDER_PATH
                file_name = f"human_conversation_{conv_number}.json"
                full_path = os.path.join(folder_path, file_name)
                with open(full_path,'w') as new_file:
                    json.dump(labeled_data,new_file)
