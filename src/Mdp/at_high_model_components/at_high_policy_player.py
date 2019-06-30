import random
from typing import List

from Mdp.at_high_model_components.at_high_model import AtHighMdpModel
from data_const import (
    JointConstants as data_consts,
    ReadableConvMetadataConstants as metadata_consts,
)
from mdp_const import MdpConsts as mdp_consts
import numpy as np

class HighPolicyPlayer:
    """
    Plays out policy assuming that high person is an agent and low person is an part of environment.
    Uses transition_counting_results to calculates probabilities.
    """

    def __init__(self, file_metadata: dict, model: AtHighMdpModel):
        """

        :param file_metadata: metadata stating who is high and who is low
        :param count_array: (4x4) probability matrix from TransitionCountingTranslator
        """
        self.model = model
        self.count_array = model.Ca
        self.file_metadata = file_metadata

    def play_policy(
        self,
        policy: np.ndarray,
        max_steps: int = 16000,
        time_step: float = 0.04,
    ) -> List[dict]:
        """
        :param file_metadata: metadata
        :param time_step:
        :param max_steps: max iterations of playing
        :param model:
        :param policy: mapping of policy[state]->action where each index of state number (0-3) and action is integer (0-1)
        :return: dictionary that looks the same as the ones in human_readible_files, where states are true to "in" or "out" but random within, namely when state is "leftEye" all it is means that it is "in", "leftEye" is randomed
        """

        result = []
        # assuming first state is None
        current_state = 0
        current_time = 0
        high_person = self.file_metadata[metadata_consts.AT_HIGH]
        low_person = self.file_metadata[metadata_consts.AT_LOW]

        for i in range(max_steps):

            action_number = policy[int(current_state)]
            new_state = self.random_next_state(current_state, action_number)
            assert (high_person != low_person), "high and low person can not be the same, something went wrong, kek"

            # gaze action are relevant only to "in" or "out" but not to the "leftEye" or "rightEye"
            if new_state == mdp_consts.NONE:

                frame = self.__create_frame(
                    high_person,
                    data_consts.OUT,
                    low_person,
                    data_consts.OUT,
                    current_time,
                    time_step,
                )
            elif new_state == mdp_consts.H_AT_L:
                frame = self.__create_frame(
                    high_person,
                    data_consts.LEFT_EYE,
                    low_person,
                    data_consts.OUT,
                    current_time,
                    time_step,
                )
            elif new_state == mdp_consts.L_AT_H:
                frame = self.__create_frame(
                    high_person,
                    data_consts.OUT,
                    low_person,
                    data_consts.LEFT_EYE,
                    current_time,
                    time_step,
                )
            elif new_state == mdp_consts.MUTUAL:
                frame = self.__create_frame(
                    high_person,
                    data_consts.LEFT_EYE,
                    low_person,
                    data_consts.LEFT_EYE,
                    current_time,
                    time_step,
                )
            else:
                raise ValueError("action is out of range, something went wrong")

            current_time += time_step

            current_state = new_state
            result.append(frame)

        return result

    def __create_frame(
            self, high_person, high_gaze, low_person, low_gaze, current_time, time_step
    ) -> dict:
        # gaze action are relevant only to "in" or "out" but not to the "leftEye" or "rightEye"
        result = {
            high_person: {
                data_consts.GAZE: high_gaze,
                data_consts.TALKING: data_consts.QUIET,
            },
            low_person: {
                data_consts.GAZE: low_gaze,
                data_consts.TALKING: data_consts.QUIET,
            },
            data_consts.TIME_START: current_time,
            data_consts.TIME_END: current_time + time_step,
            data_consts.TYPE: data_consts.DATA,
            data_consts.MAIN: high_person,
        }
        return result

    def random_next_state(self, model_current_state:int, agent_action_number:int)->int:

        cm = self.count_array

        # 0 - None 1 - A at B 2 - B at A 3 - Mutual
        # 0 - not look, 1 - look

        # 4x4, in - 1, out - 0, array[0,0]
        # i am high, low is environment

        rnd = random.random()

        list_of_possible_actions = self.model.graph[model_current_state][agent_action_number]
        last_proba = 0
        for proba, next_state in list_of_possible_actions:
            if rnd < last_proba+proba:
                return next_state
            else: last_proba+= proba

        raise ValueError("no state was chosen, something went wrong")

