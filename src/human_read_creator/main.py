import os
import json
from typing import List

from settings import Settings
from data_const import (
    UsableConversationConstants as ConstUC,
    ReadableConvMetadataConstants as ReadConst,
)
from human_read_creator.conversation_gaze_translator import (
    ConversationGazeTranslator,
    Utils,
)
from human_read_creator.at_high_at_low_calculator import AtHightAtLowCalculator

OUTPUT_FILE_NAME_PATTERN = "human_readable_conversation_{}.json"


def save_conversation_to_file(labeled_data,settings:Settings):

    folder_path = settings.HUMAN_READABLE_FOLDER_PATH
    file_name = OUTPUT_FILE_NAME_PATTERN.format(
        conv_number
    )  # f"human_conversation_{conv_number}.json"
    full_path = os.path.join(folder_path, file_name)
    with open(full_path, "w") as new_file:
        json.dump(labeled_data, new_file)


def save_at_high_calc_to_file(at_high_at_low_calculators: List[AtHightAtLowCalculator], settings: Settings):
    """
    Saves at low and at high values to file
    :param at_high_at_low_calculators: list of calculators that calculate at low and at high values
    """
    result_dict = {}

    for calc in at_high_at_low_calculators:
        file_name, person1_at_person2, person2_at_person1 = calc.get_results()
        if person1_at_person2 >= person2_at_person1:
            person_high = ReadConst.PERSON1
            person_low = ReadConst.PERSON2
        else:
            person_high = ReadConst.PERSON2
            person_low = ReadConst.PERSON1

        result_dict[file_name] = {
            ReadConst.PERSON_1_AT_PERSON_2_PERCENTAGE: person1_at_person2,
            ReadConst.PERSON_2_AT_PERSON_1_PERCENTAGE: person2_at_person1,
            ReadConst.AT_HIGH: person_high,
            ReadConst.AT_LOW: person_low,
        }


    full_path = settings.READABLE_METADATA_FILE_PATH
    with open(full_path, "w") as new_file:
        json.dump(result_dict, new_file)


if __name__ == "__main__":
    # JOINT_FILE_PATH = os.path.join(
    #     settings.JOINT_DATA_FOLDER_PATH, "conversation_04.json"
    # )
    # CLUSTER_FILE_PATH = os.path.join(
    #     settings.CLUSTER_DATA_FOLDER_PATH, "clustering_04.json"
    # )

    sett = Settings(1, 0.99, 0.001, 0.1)

    with open(sett.USABLE_CONVERSATIONS_FILE_PATH, "r") as usable_conversation_file:

        usable_conversations_data = json.loads(usable_conversation_file.read())[
            ConstUC.CONVERSATIONS
        ]

        list_of_at_high_calculators = []  # type: List[AtHightAtLowCalculator]
        for conversation_data in usable_conversations_data:
            conv_number = conversation_data[ConstUC.NUMBER]
            joint_file_path = os.path.join(
                sett.JOINT_DATA_FOLDER_PATH,
                f"conversation_{str(conv_number).zfill(2)}.json",
            )
            cluster_file_path = os.path.join(
                sett.CLUSTER_DATA_FOLDER_PATH,
                f"clustering_{str(conv_number).zfill(2)}.json",
            )

            main_person = conversation_data[ConstUC.MAIN]
            other_person = Utils.people_dict[main_person]
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
                    joint_data_parsed,
                    cluster_data_parsed,
                    main_person,
                    other_person,
                    main_person_clusters,
                    other_person_clusters,
                )
                at_high_at_low_calculator = AtHightAtLowCalculator(
                    OUTPUT_FILE_NAME_PATTERN.format(conv_number)
                )
                labeled_data = conversation_translator.convert_to_readable(
                    at_high_at_low_calculator
                )

                list_of_at_high_calculators.append(at_high_at_low_calculator)
                save_conversation_to_file(labeled_data,sett)

        save_at_high_calc_to_file(list_of_at_high_calculators,sett)
