import os

import numpy as np
from mdp_const import MdpConsts as consts
from Mdp.at_high_model_components.at_high_model import AtHighMdpModel
import settings
from Mdp.transition_counting_translator import TransitionCountingTranslator


class MdpUtils:
    __AtHighMdpModel = None  # type: AtHighMdpModel

    @staticmethod
    def get_at_high_mdp_model() -> AtHighMdpModel:
        """
        Gets MDP model from "transition_counting_results.npy" file
        :return: MDP model as graph.
        """
        if MdpUtils.__AtHighMdpModel:
            return MdpUtils.__AtHighMdpModel

        else:
            file = os.path.join(
                settings.MY_DATA_FOLDER_PATH, "transition_counting_results_with_talk.npy"
            )
            array = np.load(file)

            MdpUtils.__AtHighMdpModel = AtHighMdpModel(array)

            return MdpUtils.__AtHighMdpModel

    @staticmethod
    def get_state(high_state: int, low_state: int):
        #TODO TO BE DELETED
        """
        Returns state of given configuration
        :param high_state: gaze state of person at high, 0 - not looking, 1 - looking
        :param low_state: gaze state of person at low, 0 - not looking, 1 - looking
        :return: integer symbolising state ( 0 - None
                1 - A at B (High at Low)
                2 - B at A (Low at High)
                3 - Mutual)
        """

        if high_state == 0 and low_state == 0:
            return consts.NONE
        elif high_state == 1 and low_state == 0:
            return consts.H_AT_L
        elif high_state == 0 and low_state == 1:
            return consts.L_AT_H
        elif high_state == 1 and low_state == 1:
            return consts.MUTUAL
        else:
            raise ValueError(
                f"No combination of gaze states matches: H:{high_state}, L:{low_state}"
            )

