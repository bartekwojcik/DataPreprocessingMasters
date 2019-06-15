import os
from collections import defaultdict

import numpy as np
from settings import Settings
from Mdp.at_high_model_components.at_high_model import AtHighMdpModel


class MdpUtils:
    __AtHighMdpModel = None  # type: AtHighMdpModel

    @staticmethod
    def get_at_high_mdp_model(settings:Settings) -> AtHighMdpModel:
        """
        Gets MDP model from current (settings and mdpsettings) "transition_results" file
        :return: MDP model as graph.
        """
        if MdpUtils.__AtHighMdpModel:
            return MdpUtils.__AtHighMdpModel

        else:
            file = os.path.join(
                settings.TRANSITION_RESULTS_FOLDER_PATH, f"transition_counting_results_with_talk_"
                                                         f"{settings.TRANSITION_FRAME_STEP}_frame_"
                                                         f"{settings.TIME_SIZE}_time_size.npy"
            )
            array = np.load(file)

            MdpUtils.__AtHighMdpModel = AtHighMdpModel(array,settings)

            return MdpUtils.__AtHighMdpModel


    @staticmethod
    def q_values_to_policy(model:AtHighMdpModel, Q:defaultdict)->np.ndarray:
        """
        translates Q values to policy
        :return:
        """
        n_states = len(model.states)
        policy = np.zeros((n_states,))

        for state, actions_values in Q.items():
            state_index = model.states.index(state)

            best_action_index = np.argmax(actions_values)

            policy[state_index] = best_action_index

        return policy


