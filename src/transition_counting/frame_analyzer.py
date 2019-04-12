from data_const import JointConstants as ConstJoint
import numpy as np
from transition_counting.gaze_processor import GazeProcessor

"""
Processes frames and counts the transition states
"""


class FrameAnalyzer:
    def process_frame(self, previous_frame: dict, frame: dict) -> np.ndarray:
        """
        Study transition changes between previous and current frame
        :param previous_frame:
        :param frame:
        :return: return 2x2x2x2 matrix that translates to:
            [Other_previous_state][Other_current_state][Main_previous_state][Main_current_state]
        """

        # assuming person1 is main, and person2 is other
        result = np.zeros((2, 2, 2, 2))

        main = ConstJoint.PERSON_1
        other = ConstJoint.PERSON_2
        gaze_processor = GazeProcessor()
        current_main_gaze_state = frame[main][ConstJoint.GAZE]
        current_other_gaze_state = frame[other][ConstJoint.GAZE]

        previous_main_gaze_state = previous_frame[main][ConstJoint.GAZE]
        previous_other_gaze_state = previous_frame[other][ConstJoint.GAZE]

        result = gaze_processor.increment_matrix(
            result,
            previous_other_gaze_state,
            current_other_gaze_state,
            previous_main_gaze_state,
            current_main_gaze_state,
            1,
        )

        return result
