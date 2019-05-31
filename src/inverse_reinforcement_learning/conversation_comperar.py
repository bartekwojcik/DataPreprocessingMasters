from typing import List, Tuple
import numpy as np
import os
import settings
from Mdp.transition_counting_translator import TransitionCountingTranslator
from transition_counting.heatmap_plotter import plot_count_heatmap
from transition_counting.transition_counter import TransitionCounter


class ConversationComparer:
    """
    Compares real conversation with the reasult created by IRL algorithm and saves plots to disc
    """
    def compare_and_save_plots(
        self,
        file_name: str,
        original_conversation: List[dict],
        calculated_conversation: List[dict],
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
        calculated_results = self.__count(
            calculated_conversation, file_metadata, frame_step,result_shape
        )

        self.__save_plots(original_results, name, "original", show)
        self.__save_plots(calculated_results, name, "calculated", show)


    def __save_plots(self, results, file_name, original_or_not: str, show: bool):
        file_name_counts = os.path.join(
            settings.COMPARISON_PLOTS_FOLDER_PATH,
            f"{file_name}_{original_or_not}_plot_counts.png",
        )
        file_name_probs = os.path.join(
            settings.COMPARISON_PLOTS_FOLDER_PATH,
            f"{file_name}_{original_or_not}_plot_probs.png",
        )

        translator = TransitionCountingTranslator(results)
        probabilities_matrix = translator.transform_to_2D_probabilities_matrix()
        count_matrix = translator.transform_to_2D_count_matrix()
        plot_count_heatmap(np.round(probabilities_matrix, decimals=2), file_name_probs, show)
        plot_count_heatmap(count_matrix, file_name_counts, show)


    def __count(self, conversation, file_metadata: dict, frame_step: int, shape:Tuple) -> np.ndarray:
        result = np.zeros(shape)
        starting_points = np.arange(0, frame_step)
        counter = TransitionCounter()

        for i in starting_points:
            file_result = counter.count_transitions(
                conversation, frame_step, i, file_metadata, shape
            )
            result += file_result

        return result

