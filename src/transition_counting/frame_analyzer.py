from data_const import JointConstants as ConstJoint, ReadableConvMetadataConstants as metaConsts
import numpy as np
from transition_counting.state_processor import StateProcessor

"""
Processes frames and counts the transition states
"""


class FrameAnalyzer:
    def process_frame(self, previous_frame: dict, frame: dict, metadata: dict) -> np.ndarray:
        """
        Study transition changes between previous and current frame. High is "Person at high" and low is "Person at low" as in paper

        :param previous_frame:
        :param frame:
        :return: return 2x2x2x2x2x2x2x2 matrix that translates to:
            [Low_gaze_previous_state][Low_gaze_current_state][High_gaze_previous_state][High_gaze_current_state]
            [Low_talk_previous_state][Low_talk_current_state][High_talk_previous_state][High_talk_current_state]
        """


        result = np.zeros((2, 2, 2, 2,2, 2, 2, 2))

        person_at_high = metadata[metaConsts.AT_HIGH]
        person_at_low = metadata[metaConsts.AT_LOW]

        high = person_at_high
        low = person_at_low

        gaze_processor = StateProcessor()
        current_main_gaze_state = frame[high][ConstJoint.GAZE]
        current_other_gaze_state = frame[low][ConstJoint.GAZE]
        current_main_talk_state = frame[high][ConstJoint.TALKING]
        current_other_talk_state = frame[low][ConstJoint.TALKING]

        previous_main_gaze_state = previous_frame[high][ConstJoint.GAZE]
        previous_other_gaze_state = previous_frame[low][ConstJoint.GAZE]
        previous_main_talk_state = frame[high][ConstJoint.TALKING]
        previous_other_talk_state = frame[low][ConstJoint.TALKING]

        result = gaze_processor.increment_matrix(
            result,
            previous_other_gaze_state,
            current_other_gaze_state,
            previous_main_gaze_state,
            current_main_gaze_state,
            previous_other_talk_state,
            current_other_talk_state,
            previous_main_talk_state,
            current_main_talk_state,
            1,
        )

        return result
