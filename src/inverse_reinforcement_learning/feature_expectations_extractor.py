from typing import List, Tuple

import numpy as np

from Mdp.at_high_model_components.at_high_model import AtHighMdpModel
from Mdp.at_high_model_components.at_high_policy_player import HighPolicyPlayer
from data_const import (
    ReadableConvMetadataConstants as read_consts,
)
from transition_counting.frame_analyzer import FrameAnalyzer


class FeatureExpectationExtractor:
    def __init__(
        self,
        states: List[Tuple],
        conversation_metadata: dict,
        discount_factor,
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
        self.previous_time = 0

    def get_feature_expectations(self, data: List[dict]) -> np.ndarray:
        """
        Analyses data step by step to find out features expectations
        :param data: dictionary of the conversation
        :return: feature expectations as a np.ndarray
        """
        data_length = len(data)
        max_steps = data_length
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

    def get_random_feature_expectations(
        self, n_steps: int, model: AtHighMdpModel, policy_player: HighPolicyPlayer
    ):
        """
        Returns random feature expectations following random policy
        :param n_steps: number of frames in produced conversation
        :param model: MDP model to get number of actions
        :param policy_player:
        :return: feature expectations of following random policy
        """

        mi = np.array([0 for n in range(self.n_states)])

        n_actions = len(model.actions)
        policy = np.random.random_integers(0, n_actions - 1, size=mi.shape)

        random_conversation = policy_player.play_policy(policy, n_steps)

        return self.get_feature_expectations(random_conversation)

    def __get_current_state(self, current_frame, previous_frame) -> np.ndarray:
        """
        Get vector of current state. For instance [0,0,1,0,TIME], [0,0,0,1,TIME]
        :param current_frame:
        :return:
        """
        high_person = self.conversation_metadata[read_consts.AT_HIGH]
        low_person = self.conversation_metadata[read_consts.AT_LOW]

        current_state_vector_without_time = FrameAnalyzer.get_gaze_talk_state_vector_from_frame(
            current_frame, high_person, low_person
        )

        previous_state_vector_without_time = FrameAnalyzer.get_gaze_talk_state_vector_from_frame(
            previous_frame, high_person, low_person
        )

        current_state_vector = current_state_vector_without_time + (self.previous_time,)

        if previous_state_vector_without_time == current_state_vector_without_time:
            self.previous_time = +1
        else:
            self.previous_time = 0

        mi = self.calculate_state_vector(current_state_vector, self.states)

        return np.array(mi)

    @classmethod
    def calculate_state_vector(cls, current_state_vector: Tuple, states: List[Tuple]):
        """
        :param current_state_vector: state vector in format (high_gaze,high_talk,low_gaze,low_talk,time)
        :param states: list of all the possible states
        :return: Hot-One encoding of the state
        """
        zeros_indices = np.zeros(len(states))

        state_index = states.index(current_state_vector)
        zeros_indices[state_index] = 1
        return zeros_indices
