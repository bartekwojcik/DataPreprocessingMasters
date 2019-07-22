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

    def __init__(
        self, file_metadata: dict, model: AtHighMdpModel, epsilon_greedy: float = 0.0
    ):
        """
        Plays out policy assuming that high person is an agent and low person is an part of environment.
        :param file_metadata: metadata of tonversation that model is using
        :param model: MPD model
        :param epsilon_greedy: parameter of how often does agent chose different action than the one from the policy
        """
        self.epsilon_greedy = epsilon_greedy
        self.model = model
        self.count_array = model.Ca
        self.file_metadata = file_metadata

    def play_policy(
        self, policy: np.ndarray, max_steps: int, time_step: float = 0.04
    ) -> List[dict]:
        """

        :param time_step:
        :param max_steps: max iterations of playing (how many frames of conversation will be created)
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
            action_index = self.__get_action(policy, current_state_index)

            new_state = self.__random_next_state(current_state_index, action_index)
            assert (
                high_person != low_person
            ), "high and low person can not be the same, something went wrong, kek"

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
        self,
        high_person,
        high_gaze,
        high_person_talk,
        low_person,
        low_gaze,
        low_person_talk,
        current_time,
        time_step,
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

    def __random_next_state(
        self, current_state_index: int, agent_action_number_index: int
    ) -> Tuple[int, int, int, int]:

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

        # if we got here it means that all probabilities were 0
        if last_proba == 0:
            # go to random place
            random_int = random.randint(0, len(list_of_possible_actions) - 1)
            return list_of_possible_actions[random_int][1]

        raise ValueError("no state was chosen, something went wrong")

    def __get_action(self, policy: np.ndarray, current_state_index: int) -> int:

        if self.epsilon_greedy <= 0:
            return int(policy[int(current_state_index)])
        elif self.epsilon_greedy > 0:

            best_action_index = int(policy[int(current_state_index)])
            n_a = len(self.model.actions)
            actions_indexes = np.arange(0, n_a, 1)
            probas = np.full(actions_indexes.shape, self.epsilon_greedy / (n_a -1))
            probas[best_action_index] = 1 - self.epsilon_greedy

            selected_action_index = np.random.choice(actions_indexes, p=probas)
            return selected_action_index
