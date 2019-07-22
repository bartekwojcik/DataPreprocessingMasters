from typing import Tuple

from data_const import (
    JointConstants as ConstJoint,
    ReadableConvMetadataConstants as metaConsts,
)
import numpy as np
import mdp_const
from mdp_const import MdpConsts
from settings import Settings
from transition_counting.transition_matrix_updater import TransitionMatrixUpdater
from transition_counting.state_utils import StateUtils

"""
Processes frames and counts the transition states
"""


class FrameAnalyzer:
    def __init__(self):
        self.previous_time = 0

    @classmethod
    def get_gaze_talk_state_vector_from_frame(
        cls, frame: dict, person_at_high: str, person_at_low: str
    ) -> Tuple[int, int, int, int]:
        """
        :param frame:
        :param person_at_high: state person at high in string form , probably from human_readable_file
        :param person_at_low: state person at low in string form , probably from human_readable_file
        :return: Tuple of 4 integers stating (high_gaze, high_talk, low_gaze, low_talk)
        """
        high = person_at_high
        low = person_at_low

        high_gaze_state = frame[high][ConstJoint.GAZE]
        high_talk_state = frame[high][ConstJoint.TALKING]

        low_gaze_state = frame[low][ConstJoint.GAZE]
        low_talk_state = frame[low][ConstJoint.TALKING]

        high_gaze = StateUtils.get_gaze_id(high_gaze_state)
        high_talk = StateUtils.get_talk_id(high_talk_state)
        low_gaze = StateUtils.get_gaze_id(low_gaze_state)
        low_talk = StateUtils.get_talk_id(low_talk_state)

        return (high_gaze, high_talk, low_gaze, low_talk)

    def process_frame(
        self,
        previous_frame: dict,
        frame: dict,
        metadata: dict,
        shape: Tuple,
        state_processor: TransitionMatrixUpdater,
        max_time_frames:int
    ) -> np.ndarray:
        """
        Study transition changes between previous and current frame.
         High is "Person at high" and low is "Person at low" as in paper

        :param previous_frame:
        :param frame:
        :return: return 2x2x2x2x2x2x2x2,TIME matrix that translates to:
             previous_high_gaze_state, current_high_gaze_state,
            previous_high_talk_state, current_high_talk_state,
            previous_low_gaze_state, current_low_gaze_state,
            previous_low_talk_state, current_low_talk_state,
            MAX TIME
        """

        result = np.zeros(shape)

        person_at_high = metadata[metaConsts.AT_HIGH]
        person_at_low = metadata[metaConsts.AT_LOW]

        high = person_at_high
        low = person_at_low

        previous_state = self.get_gaze_talk_state_vector_from_frame(
            previous_frame, high, low
        )
        current_state = self.get_gaze_talk_state_vector_from_frame(frame, high, low)

        if self.previous_time == max_time_frames:
            self.previous_time = 0

        result = state_processor.increment_matrix(
            result,
            previous_state[0],
            previous_state[1],
            previous_state[2],
            previous_state[3],
            current_state[0],
            current_state[1],
            current_state[2],
            current_state[3],
            self.previous_time,
            1,
        )

        if previous_state == current_state:
            # the state is the same so we increment time step
            self.previous_time += 1
        else:
            # state changed and we get back to the state 0 of new state
            self.previous_time = 0

        return result
