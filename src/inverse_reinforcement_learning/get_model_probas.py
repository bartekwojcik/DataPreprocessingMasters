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


class ModelProbasGetter:  # lol
    def get_model_probas(
        self,
        conversation,
        file_metadata: dict,
        frame_step: int

    ) -> AtHighMdpModel:
        # just to get dimentions of the result matrix
        count_array_shape = (2,2,2,2)

        count_array = self.__count(
            conversation, file_metadata, frame_step, count_array_shape
        )

        #not really usefull
        #self.__save_plots(count_array, filename, "original", verbose,settings)

        translator = TransitionCountingTranslator(count_array)
        ca = translator.transform_to_4x4_count_matrix()

        model = AtHighMdpModel(ca)

        return model

    def __count(
        self, conversation, file_metadata: dict, frame_step: int, shape: Tuple
    ) -> np.ndarray:
        result = np.zeros(shape)
        starting_points = np.arange(0, frame_step)
        counter = TransitionCounter()

        for i in starting_points:
            file_result = counter.count_transitions(
                conversation, frame_step, i, file_metadata,
            )
            result += file_result

        return result

