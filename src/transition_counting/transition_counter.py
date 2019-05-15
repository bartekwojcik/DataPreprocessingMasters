import numpy as np
from transition_counting.frame_analyzer import FrameAnalyzer


class TransitionCounter:
    def count_transitions(self, file_data: dict, step: int, starting_step: int, metadata:dict) -> np.ndarray:
        """
        Counts transitions. High is "Person at high" and low is "Person at low" as in paper

        :return: return 2x2x2x2 matrix that translates to:
            [Low_previous_state][Low_current_state][High_previous_state][High_current_state]
        """
        result = np.zeros((2, 2, 2, 2))

        previous_frame = None

        #take only nth item at list
        for frame in file_data[starting_step::step]:
            if previous_frame is None:
                previous_frame = frame
                continue

            frame_analyzer = FrameAnalyzer()
            this_frame_result = frame_analyzer.process_frame(previous_frame, frame, metadata)
            result = result + this_frame_result

        return result
