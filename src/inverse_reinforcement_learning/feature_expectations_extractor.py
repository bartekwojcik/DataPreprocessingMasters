from typing import List

from Mdp.mdp_utils import Simple16ActionMdpModel, MdpUtils
from data_const import JointConstants as consts, ReadableConvMetadataConstants as read_consts
from transition_counting.gaze_utils import GazeUtils

import numpy as np

class FeatureExpectationExtractor:

    def __init__(self, data:dict, n_states:int, conversation_metadata:dict, max_steps:int = 10000, discount_factor = 0.95):
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
        self.data = data
        self.eye_states = np.eye(self.n_states)

    def get_experts_feature_expectations(self)-> np.ndarray:
        """
        Analyses data step by step to find out features expectations
        :return: experts feature expectations
        """
        data_length = len(self.data)
        max_steps = self.max_steps if data_length > self.max_steps else data_length
        mi = np.array([0 for n in range(self.n_states)])

        for i in range(max_steps):
            current_frame = self.data[i]
            state_vector = self.__get_current_state(current_frame)
            #http://3dvision.princeton.edu/courses/COS598/2014sp/slides/lecture07_reinforcement.pdf
            mi = mi + ((self.discount_factor ** i) * state_vector)

        # possible it has to be normalised (1/m sum(m,i=1)) on slides but i think this is done in
        # irl_algorithm_solver in __init__ (make sure that is the same part of algorithm, not two different)
        return mi

    def __get_current_state(self, frame)-> np.ndarray:
        """
        Get vector of current state. For instance [0,0,1,0] = Low at High, [0,0,0,1] = mutual etc
        :param frame:
        :return:
        """
        high_person = self.conversation_metadata[read_consts.AT_HIGH]
        low_person = self.conversation_metadata[read_consts.AT_LOW]

        high_data = frame[high_person]
        high_gaze_state = GazeUtils.get_gaze_id(high_data[consts.GAZE])

        low_data = frame[low_person]
        low_gaze_state = GazeUtils.get_gaze_id(low_data[consts.GAZE])

        state_index = MdpUtils.get_state(high_gaze_state,low_gaze_state)
        state_vector = self.eye_states[state_index]

        return np.array(state_vector)





