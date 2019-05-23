from typing import List, Tuple

import numpy as np
from transition_counting.frame_analyzer import FrameAnalyzer


class TransitionCounter:
    def count_transitions(self, file_data: List[dict], step: int, starting_step: int, metadata:dict, shape: Tuple) -> np.ndarray:
        """
        Counts transitions. High is "Person at high" and low is "Person at low" as in paper

        :return: return 2x2x2x2x2x2x2x2 matrix that translates to:
           return 2x2x2x2x2x2x2x2 matrix that translates to:
             previous_high_gaze_state, current_high_gaze_state,
            previous_high_talk_state, current_high_talk_state,
            previous_low_gaze_state, current_low_gaze_state,
            previous_low_talk_state, current_low_talk_state,
        """
        result = np.zeros(shape)

        previous_frame = None

        #take only nth item at list
        for frame in file_data[starting_step::step]:
            if previous_frame is None:
                previous_frame = frame
                continue

            frame_analyzer = FrameAnalyzer()
            this_frame_result = frame_analyzer.process_frame(previous_frame, frame, metadata, shape)
            result = result + this_frame_result

        return result
