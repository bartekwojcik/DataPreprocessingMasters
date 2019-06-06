import os
import mdp_const
import numpy as np
from mdp_const import MdpConsts as mdp
import settings
from Mdp.at_high_model_components.at_high_model import AtHighMdpModel


class MdpUtils:
    __AtHighMdpModel = None  # type: AtHighMdpModel

    @staticmethod
    def get_at_high_mdp_model() -> AtHighMdpModel:
        """
        Gets MDP model from current (settings and mdpsettings) "transition_results" file
        :return: MDP model as graph.
        """
        if MdpUtils.__AtHighMdpModel:
            return MdpUtils.__AtHighMdpModel

        else:
            file = os.path.join(
                settings.TRANSITION_RESULTS_FOLDER_PATH, f"transition_counting_results_with_talk_{settings.TRANSITION_FRAME_STEP}_frame_{mdp_const.TIME_SIZE}_time_size.npy"
            )
            array = np.load(file)

            MdpUtils.__AtHighMdpModel = AtHighMdpModel(array)

            return MdpUtils.__AtHighMdpModel

