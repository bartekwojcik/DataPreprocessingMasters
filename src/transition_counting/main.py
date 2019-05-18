import os

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import settings
import json
import numpy as np
from transition_counting.transition_counter import TransitionCounter
from transition_counting.gaze_processor import GazeProcessor
#import pylab as plt
from Mdp.transition_counting_translator import  TransitionCountingTranslator

from transition_counting.heatmap_plotter import plot_count_heatmap

def plot_heatmaps(result_array:np.ndarray)->None:
    file_name_counts = os.path.join(settings.MY_DATA_FOLDER_PATH, "heat_plot_counts.png")
    file_name_probs = os.path.join(settings.MY_DATA_FOLDER_PATH, "heat_plot_probs.png")

    translator = TransitionCountingTranslator(result_array)
    count_matrix = translator.transform_to_4x4_count_matrix()
    probabilities_matrix = translator.transform_to_4x4_probabilities_matrix()
    plot_count_heatmap(count_matrix, file_name_counts)
    plot_count_heatmap(np.round(probabilities_matrix, decimals=3),file_name_probs)

    # im = plt.matshow(results, cmap='hot', aspect='auto') # pl is pylab imported a pl
    # plt.colorbar(im)
    # plt.show()

if __name__ == "__main__":

    folder_path = settings.HUMAN_READABLE_FOLDER_PATH
    result = np.zeros((2, 2, 2, 2))
    counter = TransitionCounter()

    metadata_file_name = "human-readable-conversation-metadata.json"
    metadata_full_path = os.path.join(settings.MY_DATA_FOLDER_PATH, metadata_file_name)
    with open(metadata_full_path, "r") as meta_data_file:
        metadata = json.loads(meta_data_file.read())

        for filename in os.listdir(folder_path):
            full_file_name = os.path.join(folder_path, filename)
            file_metadata= metadata[filename]

            with open(full_file_name, "r") as data_raw:
                frame_step = 1
                starting_points = np.arange(0, frame_step)
                data = json.loads(data_raw.read())

                for i in starting_points:
                    file_result = counter.count_transitions(data, frame_step, i, file_metadata)
                    result += file_result



    gaze_processor = GazeProcessor()
    print(result)
    result_file_path = os.path.join(
        settings.MY_DATA_FOLDER_PATH, "transition_counting_results"
    )
    np.save(result_file_path, result)
    plot_heatmaps(result)
    print(f"results saved to {result_file_path}")

    # LOW IN-> OUT
    print("LOW IN-> OUT #############")
    in_out_in_in = gaze_processor.decode_matrix(result, 1, 0, 1, 1)
    print(f"in_out_in_in: {in_out_in_in }")
    in_out_in_out = gaze_processor.decode_matrix(result, 1, 0, 1, 0)
    print(f"in_out_in_out: {in_out_in_out }")
    in_out_out_out = gaze_processor.decode_matrix(result, 1, 0, 0, 0)
    print(f"in_out_out_out: {in_out_out_out }")
    in_out_out_in = gaze_processor.decode_matrix(result, 1, 0, 0, 1)
    print(f"in_out_out_in: {in_out_out_in }")

    # LOW IN-> IN
    print("LOW IN-> IN #############")
    in_in_in_in = gaze_processor.decode_matrix(result, 1, 1, 1, 1)
    print(f"in_in_in_in: {in_in_in_in }")
    in_in_in_out = gaze_processor.decode_matrix(result, 1, 1, 1, 0)
    print(f"in_in_in_out: {in_in_in_out }")
    in_in_out_out = gaze_processor.decode_matrix(result, 1, 1, 0, 0)
    print(f"in_in_out_out: {in_in_out_out }")
    in_in_out_in = gaze_processor.decode_matrix(result, 1, 1, 0, 1)
    print(f"in_in_out_in: {in_in_out_in }")

    # LOW OUT-> OUT
    print("LOW OUT-> OUT #############")
    out_out_in_in = gaze_processor.decode_matrix(result, 0, 0, 1, 1)
    print(f"out_out_in_in: {out_out_in_in }")
    out_out_in_out = gaze_processor.decode_matrix(result, 0, 0, 1, 0)
    print(f"out_out_in_out: {out_out_in_out }")
    out_out_out_out = gaze_processor.decode_matrix(result, 0, 0, 0, 0)
    print(f"out_out_out_out: {out_out_out_out }")
    out_out_out_in = gaze_processor.decode_matrix(result, 0, 0, 0, 1)
    print(f"out_out_out_in: {out_out_out_in }")

    # LOW OUT-> IN
    print("LOW OUT-> IN #############")
    out_in_in_in = gaze_processor.decode_matrix(result, 0, 1, 1, 1)
    print(f"out_in_in_in: {out_in_in_in }")
    out_in_in_out = gaze_processor.decode_matrix(result, 0, 1, 1, 0)
    print(f"out_in_in_out: {out_in_in_out }")
    out_in_out_out = gaze_processor.decode_matrix(result, 0, 1, 0, 0)
    print(f"out_in_out_out: {out_in_out_out }")
    out_in_out_in = gaze_processor.decode_matrix(result, 0, 1, 0, 1)
    print(f"out_in_out_in: {out_in_out_in }")
