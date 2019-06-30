from typing import List, Tuple
import numpy as np
import os
from Mdp.transition_counting_translator import TransitionCountingTranslator
from transition_counting.heatmap_plotter import plot_count_heatmap
from transition_counting.transition_counter import TransitionCounter
import settings

class ConversationComparer:
    """
    Compares real conversation with the reasult created by IRL algorithm and saves plots to disc
    """
    def compare_and_save_plots(
        self,
        file_name: str,
        original_conversation: List[dict],
        calculated_conversations: List[List[dict]],
        t_values: List[float],
        frame_step: int,
        file_metadata: dict,
        show: bool,
        result_shape: Tuple
    ):
        base = os.path.basename(file_name)
        # get file name without extention
        name = os.path.splitext(base)[0]
        original_results = self.__count(
            original_conversation, file_metadata, frame_step,result_shape
        )

        self.__save_plots(original_results, name, "original", show)

        for i, conversation in enumerate(calculated_conversations):
            calculated_results = self.__count(
                conversation, file_metadata, frame_step, result_shape
            )

            t = np.round(t_values[i], decimals=2)

            self.__save_plots(calculated_results, name, "calculated", show=False, t_value=t)


    def __save_plots(self, results, file_name, original_or_not: str, show: bool, t_value:float=None):

        folder_name = f"just_gaze"

        folder_path = os.path.join(
            settings.COMPARISON_PLOTS_FOLDER_PATH, folder_name
        )

        if not os.path.exists(folder_path):
            os.mkdir(folder_path)



        if t_value == None:
            file_name_probs = os.path.join(
                settings.COMPARISON_PLOTS_FOLDER_PATH, folder_path,
                f"{file_name}_{original_or_not}_plot_probs.png",
            )
            file_name_counts = os.path.join(
                settings.COMPARISON_PLOTS_FOLDER_PATH, folder_path,
                f"{file_name}_{original_or_not}_plot_counts.png",
            )
        else:
            file_name_probs = os.path.join(
                settings.COMPARISON_PLOTS_FOLDER_PATH, folder_path,
                f"{file_name}_t_{t_value}_{original_or_not}_plot_probs.png",
            )
            file_name_counts = os.path.join(
                settings.COMPARISON_PLOTS_FOLDER_PATH, folder_path,
                f"{file_name}_t_{t_value}_{original_or_not}_plot_counts.png",
            )

        translator = TransitionCountingTranslator(results)
        probabilities_matrix = translator.transform_to_4x4_probabilities_matrix()
        count_matrix = translator.transform_to_4x4_count_matrix()
        plot_count_heatmap(np.round(probabilities_matrix, decimals=2), file_name_probs, show)
        plot_count_heatmap(count_matrix, file_name_counts, show)


    def __count(self, conversation, file_metadata: dict, frame_step: int, shape:Tuple) -> np.ndarray:
        result = np.zeros(shape)
        starting_points = np.arange(0, frame_step)
        counter = TransitionCounter()

        for i in starting_points:
            file_result = counter.count_transitions(
                conversation, frame_step, i, file_metadata
            )
            result += file_result

        return result

