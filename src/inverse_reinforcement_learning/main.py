from settings import Settings
import os
import json
from inverse_reinforcement_learning.process_file import process_file


def main_synchronous(settings: Settings, VERBOSE: bool, root_folder_for_plots:str):
    HUMAN_READABLE_FOLDER_PATH = settings.HUMAN_READABLE_FOLDER_PATH
    METADATA_PATH = settings.READABLE_METADATA_FILE_PATH


    with open(METADATA_PATH, "r") as metadata_file:
        metadata_json = json.loads(metadata_file.read())

        for filename in os.listdir(HUMAN_READABLE_FOLDER_PATH):
            full_file_name = os.path.join(HUMAN_READABLE_FOLDER_PATH, filename)

            print(f"file started: {full_file_name} ")

            with open(full_file_name, "r") as conversation_file:
                conv_json = json.loads(conversation_file.read())

                heatmap_folder_path = os.path.join(root_folder_for_plots,filename,"heatmaps")
                policies_save_folder_path = os.path.join(root_folder_for_plots,filename,"policies")

                process_file(metadata_json, filename,
                             conv_json, full_file_name,
                             VERBOSE,
                             16000,
                             settings.TRANSITION_FRAME_STEP,
                             settings.TIME_SIZE,
                             Q_ITERATIONS=5,
                             Q_ALPHA=0.5,Q_EPSILON=0.05,IRL_SOLVER_EPSILON=0.5,
                             heatmap_folder_path=heatmap_folder_path
                             ,policies_save_folder_path=policies_save_folder_path
                             ,DISCOUNT_FACTOR= 0.9999999
                             )




if __name__ == "__main__":
    VERBOSE = True
    settings = Settings()

    root_folder_for_plots = settings.MY_DATA_FOLDER_PATH
    main_synchronous(settings,VERBOSE,root_folder_for_plots)

