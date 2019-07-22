from settings import Settings
import os
import json
from inverse_reinforcement_learning.process_file import process_file


def main_synchronous(
    verbose: bool,
    root_folder_for_plots: str,
    human_readable_folder_path,
    metadata_path,
    transition_frame_step,
    time_size
):
    """
    performs IRL for all files in human_readable_folder_path complemented with file metadata_path
    DONT FORGET TO CHANGE Q LEARNING VALUES INSIDE (SORRY!)
    Q_ITERATIONS=5,
    Q_ALPHA=0.5,
    Q_EPSILON=0.05,
    IRL_SOLVER_EPSILON=0.5,
    DISCOUNT_FACTOR=0.9999999,
    :param verbose: shows or hides intermediate plots during the program run
    :param root_folder_for_plots: folder to store all the results stored per conversation
    :param human_readable_folder_path: path to folder that contains human_readable_conversation files that will be processed
    :param metadata_path: path to file with metadata (originally "human-readable-conversation-metadata.json")
    :param transition_frame_step: if 1 it checks frame by frame, if 2, every second frame are "neighbouring", does not lose any data
    :param time_size: how many frames are the maximum amount of frames before staying in the same state restarts time countdown to zero
    :return: heatmaps and numpy files with policies, rewards etc stored in root_folder_for_plots. Each np.y file is a list of it's values
            (so file ending with "_policies.npy" with have 50 arrays and each of them will be a list of action per state)
            (a file of "T_values.npy" will be a list of 50 t vales per each conversation following each policy)

            so first value from "human_readable_conversation_11.json_T_values.npy" is a value of conversation that followed first policy from
            "human_readable_conversation_11.json_policies.npy" that was created from first Q values from "human_readable_conversation_11.json_Q_values.npy"
            and so on for rewards per state of "human_readable_conversation_11.json_rewards.npy" and IRL weights in "human_readable_conversation_11.json_w_values.npy"
            where 50 is a example number of IRL_solver_iteration


    """

    with open(metadata_path, "r") as metadata_file:
        metadata_json = json.loads(metadata_file.read())

        for filename in os.listdir(human_readable_folder_path):
            full_file_name = os.path.join(human_readable_folder_path, filename)

            print(f"file started: {full_file_name} ")

            with open(full_file_name, "r") as conversation_file:
                conv_json = json.loads(conversation_file.read())

                heatmap_folder_path = os.path.join(
                    root_folder_for_plots, filename, "heatmaps"
                )
                policies_save_folder_path = os.path.join(
                    root_folder_for_plots, filename, "policies"
                )

                process_file(
                    metadata_json,
                    filename,
                    conv_json,
                    full_file_name,
                    verbose,
                    16,  # 16000
                    transition_frame_step,
                    time_size,
                    Q_ITERATIONS=5,
                    Q_ALPHA=0.5,
                    Q_EPSILON=0.05,
                    DISCOUNT_FACTOR=0.9999999,
                    heatmap_folder_path=heatmap_folder_path,
                    policies_save_folder_path=policies_save_folder_path,
                    IRL_SOLVER_ITERATIONS = 50,
                    IRL_SOLVER_EPSILON=0.5,
                )


if __name__ == "__main__":
    VERBOSE = False
    settings = Settings()

    root_folder_for_plots = settings.COMPARISON_PLOTS_FOLDER_PATH
    HUMAN_READABLE_FOLDER_PATH = settings.HUMAN_READABLE_FOLDER_PATH
    METADATA_PATH = settings.READABLE_METADATA_FILE_PATH
    TRANSITION_FRAME_STEP = settings.TRANSITION_FRAME_STEP
    TIME_SIZE = settings.TIME_SIZE

    main_synchronous(
        VERBOSE,
        root_folder_for_plots,
        HUMAN_READABLE_FOLDER_PATH,
        METADATA_PATH,
        TRANSITION_FRAME_STEP,
        TIME_SIZE
    )
