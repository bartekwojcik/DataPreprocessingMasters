import random
from typing import List, Tuple

import settings
from Mdp.at_high_model_components.at_high_model import AtHighMdpModel
from Mdp.at_high_model_components.at_high_policy_player import HighPolicyPlayer
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
        states: List[Tuple],
        conversation_metadata: dict,
        max_steps: int = 20000,
        discount_factor=settings.DISCOUNT_FACTOR,
    ):
        """
        Will "play out" Markov chain / MDP to get feature expectations
        :param data: dict of conversation data
        :param n_states: number of possible states in the model
        :param conversation_metadata: metadata stating who is high and who is low
        :param max_steps: maximal amount of steps model will check or perform
        :param discount_factor: gamma
        """
        self.n_states = len(states)
        self.states = states
        self.discount_factor = discount_factor
        self.conversation_metadata = conversation_metadata
        self.max_steps = max_steps
        self.previous_time = 0


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

    def get_random_feature_expectations(self,n_steps:int, model:AtHighMdpModel, policy_player:HighPolicyPlayer):

        mi = np.array([0 for n in range(self.n_states)])

        n_actions = len(model.actions)
        policy = np.random.random_integers(0,n_actions-1,size=mi.shape)

        random_conversation = policy_player.play_policy(policy,n_steps)

        return self.get_feature_expectations(random_conversation)


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

        current_state_vector = current_state_vector_without_time + (self.previous_time,)

        if previous_state_vector_without_time == current_state_vector_without_time:
            self.previous_time = +1
        else:
            self.previous_time = 0


        mi = self.calculate_state_vector(current_state_vector,self.states)

        return np.array(mi)

    @classmethod
    def calculate_state_vector(cls, current_state_vector:Tuple, states: List[Tuple]):
        zeros_indices = np.zeros(len(states))

        state_index = states.index(current_state_vector)
        zeros_indices[state_index] = 1
        return zeros_indices


