from typing import Tuple

from Mdp.at_high_model_components.at_high_model import AtHighMdpModel
from Mdp.mdp_utils import MdpUtils
from transition_counting.transition_counter import TransitionCounter
import numpy as np
from typing import Tuple


class ModelProbasGetter:#lol

    def get_model_probas(self,conversation, file_metadata: dict, frame_step: int,)->AtHighMdpModel:
        #just to get dimentions of the result matrix
        count_array_shape = MdpUtils.get_at_high_mdp_model().Ca.shape

        count_array = self.__count(conversation,file_metadata,frame_step,count_array_shape)

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


