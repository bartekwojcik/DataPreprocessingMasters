from typing import List, Tuple
import numpy as np
import os
from settings import Settings
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
        result_shape: Tuple,
        settings: Settings
    ):
        base = os.path.basename(file_name)
        # get file name without extention
        name = os.path.splitext(base)[0]
        original_results = self.__count(
            original_conversation, file_metadata, frame_step,result_shape,settings
        )
        calculated_results = self.__count(
            calculated_conversation, file_metadata, frame_step,result_shape,settings
        )



        self.__save_plots(original_results, name, "original", show,settings)
        self.__save_plots(calculated_results, name, "calculated", show,settings)


    def __save_plots(self, results, file_name, original_or_not: str, show: bool, settings:Settings):

        folder_name = f"frame_{settings.TRANSITION_FRAME_STEP}_time_{settings.TIME_SIZE}"

        folder_path = os.path.join(
            settings.HISTOGRAMS_FOLDER_PATH, folder_name
        )

        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        file_name_counts = os.path.join(
            settings.COMPARISON_PLOTS_FOLDER_PATH, folder_path,
            f"{settings.GLOBAL_PREFIX_FOR_FILE_NAMES}_{file_name}_{original_or_not}_plot_counts.png",
        )
        file_name_probs = os.path.join(
            settings.COMPARISON_PLOTS_FOLDER_PATH, folder_path,
            f"{settings.GLOBAL_PREFIX_FOR_FILE_NAMES}_{file_name}_{original_or_not}_plot_probs.png",
        )

        translator = TransitionCountingTranslator(results,settings)
        probabilities_matrix = translator.transform_to_2D_probabilities_matrix()
        count_matrix = translator.transform_to_2D_count_matrix()
        plot_count_heatmap(np.round(probabilities_matrix, decimals=2), file_name_probs, show)
        plot_count_heatmap(count_matrix, file_name_counts, show)


    def __count(self, conversation, file_metadata: dict, frame_step: int, shape:Tuple,settings: Settings) -> np.ndarray:
        result = np.zeros(shape)
        starting_points = np.arange(0, frame_step)
        counter = TransitionCounter()

        for i in starting_points:
            file_result = counter.count_transitions(
                conversation, frame_step, i, file_metadata, shape,settings
            )
            result += file_result

        return result

