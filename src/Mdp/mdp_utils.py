import os

import numpy as np

import settings
from Mdp.at_high_model_components.at_high_model import AtHighMdpModel


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
                settings.MY_DATA_FOLDER_PATH, f"transition_counting_results_with_talk_{settings.TRANSITION_FRAME_STEP}_frame.npy"
            )
            array = np.load(file)

            MdpUtils.__AtHighMdpModel = AtHighMdpModel(array)

            return MdpUtils.__AtHighMdpModel

