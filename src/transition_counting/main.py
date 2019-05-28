import os
from mdp_const import MdpConsts as mdp
import settings
import json
import numpy as np
from transition_counting.transition_counter import TransitionCounter
from transition_counting.state_processor import StateProcessor
#import pylab as plt
from Mdp.transition_counting_translator import  TransitionCountingTranslator

from transition_counting.heatmap_plotter import plot_count_heatmap

def plot_heatmaps(result_array:np.ndarray, file_name_counts:str, file_name_probas:str)->None:
    file_name_counts = os.path.join(settings.MY_DATA_FOLDER_PATH,file_name_counts )
    file_name_probs = os.path.join(settings.MY_DATA_FOLDER_PATH, file_name_probas)

    translator = TransitionCountingTranslator(result_array)
    count_matrix = translator.transform_to_2D_count_matrix()
    probabilities_matrix = translator.transform_to_2D_probabilities_matrix()
    plot_count_heatmap(count_matrix, file_name_counts)
    plot_count_heatmap(np.round(probabilities_matrix, decimals=3),file_name_probs)



if __name__ == "__main__":
    """
    the time transitions are from perspective of high, namely: after what time did the at high person change his state 
    
    """

    folder_path = settings.HUMAN_READABLE_FOLDER_PATH
    time_size = mdp.TIME_SIZE
    global_results = np.zeros((2, 2,time_size, 2, 2,time_size, 2, 2,time_size, 2, 2,time_size))
    counter = TransitionCounter()


    metadata_full_path = settings.READABLE_METADATA_FILE_PATH
    with open(metadata_full_path, "r") as meta_data_file:
        metadata = json.loads(meta_data_file.read())

        for filename in os.listdir(folder_path):
            full_file_name = os.path.join(folder_path, filename)
            file_metadata= metadata[filename]
            file_results = np.zeros_like(global_results)
            base = os.path.basename(filename)
            clear_name = os.path.splitext(base)[0]
            with open(full_file_name, "r") as data_raw:
                FRAME_STEP = settings.TRANSITION_FRAME_STEP
                starting_points = np.arange(0, FRAME_STEP)
                data = json.loads(data_raw.read())

                for i in starting_points:
                    one_result = counter.count_transitions(data, FRAME_STEP, i, file_metadata, global_results.shape)
                    global_results += one_result
                    file_results += one_result

                #plot_heatmaps(file_results, f"{clear_name}_counts.jpg", f"{clear_name}_probas.jpg")
                debug = 5


    print(global_results)
    result_file_path = os.path.join(
        settings.MY_DATA_FOLDER_PATH, "transition_counting_results_with_talk"
    )
    np.save(result_file_path, global_results)
    plot_heatmaps(global_results,"heat_plot_counts.png","heat_plot_probs.png")
    print(f"results saved to {result_file_path}")


