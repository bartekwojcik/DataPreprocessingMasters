from typing import Tuple

from data_const import JointConstants as ConstJoint, ReadableConvMetadataConstants as metaConsts
import numpy as np
from transition_counting.state_processor import StateProcessor

"""
Processes frames and counts the transition states
"""


class FrameAnalyzer:

    def process_frame(self, previous_frame: dict, frame: dict, metadata: dict, shape: Tuple, state_processor: StateProcessor) -> np.ndarray:
        """
        Study transition changes between previous and current frame. High is "Person at high" and low is "Person at low" as in paper

        :param previous_frame:
        :param frame:
        :return: return 2x2x2x2x2x2x2x2 matrix that translates to:
             previous_high_gaze_state, current_high_gaze_state,
            previous_high_talk_state, current_high_talk_state,
            previous_low_gaze_state, current_low_gaze_state,
            previous_low_talk_state, current_low_talk_state,
        """

        result = np.zeros(shape)

        person_at_high = metadata[metaConsts.AT_HIGH]
        person_at_low = metadata[metaConsts.AT_LOW]

        high = person_at_high
        low = person_at_low


        previous_high_gaze_state = previous_frame[high][ConstJoint.GAZE]
        current_high_gaze_state = frame[high][ConstJoint.GAZE]
        previous_high_talk_state = previous_frame[high][ConstJoint.TALKING]
        current_high_talk_state = frame[high][ConstJoint.TALKING]

        previous_low_gaze_state = previous_frame[low][ConstJoint.GAZE]
        current_low_gaze_state = frame[low][ConstJoint.GAZE]
        previous_low_talk_state = previous_frame[low][ConstJoint.TALKING]
        current_low_talk_state = frame[low][ConstJoint.TALKING]

        #unfortunately this metod also takes care of checking if states were the same and updates time.
        #sorry future you
        result = state_processor.increment_matrix(
            result,
            previous_high_gaze_state, current_high_gaze_state,
            previous_high_talk_state, current_high_talk_state,
            previous_low_gaze_state, current_low_gaze_state,
            previous_low_talk_state, current_low_talk_state,
            1,
        )

        return result
