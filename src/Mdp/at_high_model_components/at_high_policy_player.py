import random
from typing import List, Tuple

import numpy as np

from Mdp.at_high_model_components.at_high_model import AtHighMdpModel
from data_const import (
    JointConstants as data_consts,
    ReadableConvMetadataConstants as metadata_consts,
)
from transition_counting.state_utils import StateUtils


class HighPolicyPlayer:
    """
    Plays out policy assuming that high person is an agent and low person is an part of environment.
    Uses transition_counting_results to calculates probabilities.
    """

    def __init__(self, file_metadata: dict, model: AtHighMdpModel):
        """

        :param file_metadata: metadata stating who is high and who is low
        """
        self.model = model
        self.count_array = model.Ca
        self.file_metadata = file_metadata

    def play_policy(
            self,
            policy: np.ndarray,
            max_steps: int = 1000,
            time_step: float = 0.04,
    ) -> List[dict]:
        """
        :param file_metadata: metadata
        :param time_step:
        :param max_steps: max iterations of playing
        :param model:
        :param policy: A policy is a mapping where policy[state]->action where state index is given state
        0 - (not look, not talk)
        1 - (not look, talk)
        2 - (look, not talk)
        3 - (look, talk)
        these indices corresponds to model.actions[index]
        :return: dictionary that looks the same as the ones in human_readible_files, where states are true to "in" or "out" but random within, namely when state is "leftEye" all it is means that it is "in", "leftEye" is randomed
        """

        result = []
        # assuming first state is None
        current_state_index = 0
        current_time = 0
        high_person = self.file_metadata[metadata_consts.AT_HIGH]
        low_person = self.file_metadata[metadata_consts.AT_LOW]

        for i in range(max_steps):
            action_index = int(policy[int(current_state_index)])
            # new_state is a tuple of length 4
            new_state = self.__random_next_state(current_state_index, action_index)
            assert (high_person != low_person), "high and low person can not be the same, something went wrong, kek"
            # TODO WORK HERE
            # gaze action are relevant only to "in" or "out" but not to the "leftEye" or "rightEye"

            high_gaze = StateUtils.gaze_id_to_string(new_state[0])  # type:str
            high_talk = StateUtils.talk_id_to_string(new_state[1])  # type:str
            low_gaze = StateUtils.gaze_id_to_string(new_state[2])  # type:str
            low_talk = StateUtils.talk_id_to_string(new_state[3])  # type:str

            frame = self.__create_frame(
                high_person,
                high_gaze,
                high_talk,
                low_person,
                low_gaze,
                low_talk,
                current_state_index,
                time_step,
            )

            current_time += time_step

            current_state_index = self.model.states.index(new_state)
            result.append(frame)

        return result

    def __create_frame(
            self, high_person, high_gaze, high_person_talk, low_person, low_gaze, low_person_talk, current_time,
            time_step
    ) -> dict:
        # gaze action are relevant only to "in" or "out" but not to the "leftEye" or "rightEye"
        result = {
            high_person: {
                data_consts.GAZE: high_gaze,
                data_consts.TALKING: high_person_talk,
            },
            low_person: {
                data_consts.GAZE: low_gaze,
                data_consts.TALKING: low_person_talk,
            },
            data_consts.TIME_START: current_time,
            data_consts.TIME_END: current_time + time_step,
            data_consts.TYPE: data_consts.DATA,
            data_consts.MAIN: high_person,
        }
        return result

    def __random_next_state(self, current_state_index: int, agent_action_number_index: int) -> Tuple[int, int, int, int]:

        # TODO WORK HERE

        cm = self.count_array

        # 0 - None 1 - A at B 2 - B at A 3 - Mutual
        # 0 - not look, 1 - look

        # 4x4, in - 1, out - 0, array[0,0]
        # i am high, low is environment

        rnd = random.random()
        current_state = self.model.states[current_state_index]
        agent_action = self.model.actions[agent_action_number_index]

        list_of_possible_actions = self.model.graph[current_state][agent_action]
        last_proba = 0
        for proba, next_state in list_of_possible_actions:
            if rnd < last_proba + proba:
                return next_state
            else:
                last_proba += proba
        #if we got here it means that all probabilities were 0
        if last_proba == 0:
            #go to random place
            random_int = random.randint(0,len(list_of_possible_actions)-1)
            return list_of_possible_actions[random_int][1]

        raise ValueError("no state was chosen, something went wrong")
