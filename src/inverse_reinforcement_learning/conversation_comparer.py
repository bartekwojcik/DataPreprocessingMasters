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
        calculated_conversations: List[List[dict]],
        t_values: List[float],
        frame_step: int,
        file_metadata: dict,
        show: bool,
        result_shape: Tuple,
            max_time_frames:int,
            heatmap_folder_path:str
    ):

        original_results = self.__count(
            original_conversation, file_metadata, frame_step, result_shape, max_time_frames
        )

        base = os.path.basename(file_name)
        # get file name without extention
        name = os.path.splitext(base)[0]


        self.__save_plots(original_results, name, "original", show,heatmap_folder_path,max_time_frames)

        for i, conversation in enumerate(calculated_conversations):
            calculated_results = self.__count(
                conversation, file_metadata, frame_step, result_shape, max_time_frames
            )

            t = np.round(t_values[i], decimals=2)

            self.__save_plots(calculated_results, name, "calculated", False, heatmap_folder_path, max_time_frames, t_value=t)


    def __save_plots(self, results,
                     file_name,
                     original_or_not: str,
                     show: bool,
                     folder_path:str,
                     max_time_frames:int,
                     t_value:float=None):

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if t_value == None:
            file_name_probs = os.path.join(
                 folder_path,
                f"{file_name}_{original_or_not}_plot_probs.png",
            )
            file_name_counts = os.path.join(
                 folder_path,
                f"{file_name}_{original_or_not}_plot_counts.png",
            )
        else:
            file_name_probs = os.path.join(
                 folder_path,
                f"t_{t_value}_{file_name}_{original_or_not}_plot_probs.png",
            )
            file_name_counts = os.path.join(
                folder_path,
                f"t_{t_value}_{file_name}_{original_or_not}_plot_counts.png",
            )

        translator = TransitionCountingTranslator(results,max_time_frames)
        probabilities_matrix = translator.transform_to_2D_probabilities_matrix()
        count_matrix = translator.transform_to_2D_count_matrix()
        plot_count_heatmap(np.round(probabilities_matrix, decimals=2), file_name_probs, show)
        plot_count_heatmap(count_matrix, file_name_counts, show)


    def __count(self, conversation, file_metadata: dict, frame_step: int, shape:Tuple,max_time_frames:int) -> np.ndarray:
        result = np.zeros(shape)
        starting_points = np.arange(0, frame_step)
        counter = TransitionCounter()

        for i in starting_points:
            file_result = counter.count_transitions(
                conversation, frame_step, i, file_metadata, shape,max_time_frames
            )
            result += file_result

        return result

