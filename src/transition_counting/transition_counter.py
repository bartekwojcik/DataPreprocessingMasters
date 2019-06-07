from typing import List, Tuple

import numpy as np

from settings import Settings
from transition_counting.frame_analyzer import FrameAnalyzer
from transition_counting.transition_matrix_updater import TransitionMatrixUpdater


class TransitionCounter:
    def count_transitions(
        self,
        file_data: List[dict],
        step: int,
        starting_step: int,
        metadata: dict,
        shape: Tuple,
        settings: Settings
    ) -> np.ndarray:
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
        state_processor = TransitionMatrixUpdater()
        frame_analyzer = FrameAnalyzer()
        # take only nth item at list
        for frame in file_data[starting_step::step]:
            if previous_frame is None:
                previous_frame = frame
                continue

            this_frame_result = frame_analyzer.process_frame(
                previous_frame, frame, metadata, shape, state_processor, settings
            )
            result = result + this_frame_result
            previous_frame = frame

        return result
