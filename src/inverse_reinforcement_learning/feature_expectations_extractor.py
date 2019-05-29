from typing import List

from mdp_const import MdpConsts
from data_const import (
    JointConstants as consts,
    ReadableConvMetadataConstants as read_consts,
)
from transition_counting.state_utils import StateUtils

import numpy as np


class FeatureExpectationExtractor:
    def __init__(
        self,
        n_states: int,
        conversation_metadata: dict,
        max_steps: int = 10000,
        discount_factor=0.95,
    ):
        """
        Will "play out" Markov chain / MDP to get feature expectations
        :param data: dict of conversation data
        :param n_states: number of possible states in the model
        :param conversation_metadata: metadata stating who is high and who is low
        :param max_steps: maximal amount of steps model will check or perform
        :param discount_factor: gamma
        """
        self.n_states = n_states
        self.discount_factor = discount_factor
        self.conversation_metadata = conversation_metadata
        self.max_steps = max_steps
        self.previous_time = 0
        self.time_denominator = MdpConsts.TIME_SIZE

    def get_experts_feature_expectations(self, data: List[dict]) -> np.ndarray:
        """
        Analyses data step by step to find out features expectations
        :return: experts feature expectations
        """
        data_length = len(data)
        max_steps = self.max_steps if data_length > self.max_steps else data_length
        mi = np.array([0 for n in range(self.n_states)])
        previous_frame = None
        for i in range(max_steps):
            current_frame = data[i]

            if not previous_frame:
                previous_frame = current_frame
                continue

            state_vector = self.__get_current_state(current_frame, previous_frame)
            # http://3dvision.princeton.edu/courses/COS598/2014sp/slides/lecture07_reinforcement.pdf
            mi = mi + ((self.discount_factor ** i) * state_vector)

            previous_frame = current_frame

        # possible it has to be normalised (1/m sum(m,i=1)) on slides but i think this is done in
        # irl_algorithm_solver in __init__ (make sure that is the same part of algorithm, not two different)
        return mi

    def __get_current_state(self, current_frame, previous_frame) -> np.ndarray:
        """
        Get vector of current state. For instance [0,0,1,0] = Low at High, [0,0,0,1] = mutual etc
        :param current_frame:
        :return:
        """
        high_person = self.conversation_metadata[read_consts.AT_HIGH]
        low_person = self.conversation_metadata[read_consts.AT_LOW]

        high_data = current_frame[high_person]
        high_gaze_state = StateUtils.get_gaze_id(high_data[consts.GAZE])
        high_talk_state = StateUtils.get_talk_id(high_data[consts.TALKING])

        low_data = current_frame[low_person]
        low_gaze_state = StateUtils.get_gaze_id(low_data[consts.GAZE])
        low_talk_state = StateUtils.get_talk_id(low_data[consts.TALKING])

        current_state_vector = (
            high_gaze_state,
            high_talk_state,
            low_gaze_state,
            low_talk_state,
            self.previous_time / self.time_denominator,
        )

        previous_high_data = previous_frame[high_person]
        previous_high_gaze_state = StateUtils.get_gaze_id(
            previous_high_data[consts.GAZE]
        )
        previous_high_talk_state = StateUtils.get_talk_id(
            previous_high_data[consts.TALKING]
        )

        previous_low_data = previous_frame[low_person]
        previous_low_gaze_state = StateUtils.get_gaze_id(previous_low_data[consts.GAZE])
        previous_low_talk_state = StateUtils.get_talk_id(
            previous_low_data[consts.TALKING]
        )

        previous_state_vector = (
            previous_high_gaze_state,
            previous_high_talk_state,
            previous_low_gaze_state,
            previous_low_talk_state,
        )

        current_state_vector_witout_time = current_state_vector[:-1]
        if previous_state_vector == current_state_vector_witout_time:
            self.previous_time = +1
        else:
            self.previous_time = 0

        return np.array(current_state_vector)
