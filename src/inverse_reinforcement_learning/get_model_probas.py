from typing import Tuple
import os
import settings
from Mdp.at_high_model_components.at_high_model import AtHighMdpModel
from Mdp.mdp_utils import MdpUtils
from Mdp.transition_counting_translator import TransitionCountingTranslator
from transition_counting.heatmap_plotter import plot_count_heatmap
from transition_counting.transition_counter import TransitionCounter
import numpy as np
from typing import Tuple


class ModelProbasGetter:#lol

    def get_model_probas(self,conversation, file_metadata: dict, frame_step: int,filename:str,verbose:bool)->AtHighMdpModel:
        #just to get dimentions of the result matrix
        count_array_shape = MdpUtils.get_at_high_mdp_model().Ca.shape

        count_array = self.__count(conversation,file_metadata,frame_step,count_array_shape)

        #self.__save_plots(count_array,filename,"original",verbose)


        model = AtHighMdpModel(count_array)

        return model



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

    def __save_plots(self, results, file_name, original_or_not: str, show: bool):
        file_name_counts = os.path.join(
            settings.COMPARISON_PLOTS_FOLDER_PATH,
            f"{settings.GLOBAL_PREFIX_FOR_FILE_NAMES}_{file_name}_{original_or_not}_plot_counts_FROM_BEFOREHAND_TO_CHECK_NEW_FUNCTION.png",
        )
        file_name_probs = os.path.join(
            settings.COMPARISON_PLOTS_FOLDER_PATH,
            f"{settings.GLOBAL_PREFIX_FOR_FILE_NAMES}_{file_name}_{original_or_not}_plot_probs_FROM_BEFOREHAND_TO_CHECK_NEW_FUNCTION.png",
        )

        translator = TransitionCountingTranslator(results)
        probabilities_matrix = translator.transform_to_2D_probabilities_matrix()
        count_matrix = translator.transform_to_2D_count_matrix()
        plot_count_heatmap(np.round(probabilities_matrix, decimals=3), file_name_probs, show)
        plot_count_heatmap(count_matrix, file_name_counts, show)


