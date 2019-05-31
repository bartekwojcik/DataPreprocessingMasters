from typing import List

from mdp_const import MdpConsts
from data_const import (
    JointConstants as consts,
    ReadableConvMetadataConstants as read_consts,
)
from transition_counting.frame_analyzer import FrameAnalyzer
from transition_counting.state_utils import StateUtils

import numpy as np


class FeatureExpectationExtractor:
    def __init__(
        self,
        n_states: int,
        conversation_metadata: dict,
        max_steps: int = 20000,
        discount_factor=0.99,
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
        self.number_of_non_time_states = len(MdpConsts.GET_TALK_AND_LOOK_STATES())

    def get_feature_expectations(self, data: List[dict]) -> np.ndarray:
        """
        Analyses data step by step to find out features expectations
        :return: feature expectations
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

    def get_random_feature_expectations(self,n_steps:int):


        # we are simulating here that agent knows only policy that asks him to go to these two random places

        first_random_state = np.array([0 for n in range(self.n_states)])
        second_random_state = np.array([1/self.number_of_non_time_states for n in range(self.n_states)])
        second_random_state[-1] = 0

        def get_the_second_one(one_state):

            if np.array_equal(one_state,first_random_state):
                return second_random_state
            else: return first_random_state

        mi = np.array([0 for n in range(self.n_states)])
        return np.array([1 for n in range(self.n_states)])
        previous_state = first_random_state
        for i in range(n_steps):

            state_vector = get_the_second_one(previous_state)
            previous_state = state_vector
            mi = mi + ((self.discount_factor ** i) * state_vector)



        return mi

    def __get_current_state(self, current_frame, previous_frame) -> np.ndarray:
        """
        Get vector of current state. For instance [0,0,1,0,TIME], [0,0,0,1,,TIME]
        :param current_frame:
        :return:
        """
        high_person = self.conversation_metadata[read_consts.AT_HIGH]
        low_person = self.conversation_metadata[read_consts.AT_LOW]

        current_state_vector_without_time = FrameAnalyzer.\
            get_gaze_talk_state_vector_from_frame(current_frame,high_person,low_person)


        previous_state_vector_without_time = FrameAnalyzer.\
            get_gaze_talk_state_vector_from_frame(previous_frame,high_person,low_person)


        if previous_state_vector_without_time == current_state_vector_without_time:
            self.previous_time = +1
        else:
            self.previous_time = 0


        current_state_vector_without_time = tuple(n/self.number_of_non_time_states for n in current_state_vector_without_time)

        current_state_vector = current_state_vector_without_time + (self.previous_time / self.time_denominator,)
        return np.array(current_state_vector)
