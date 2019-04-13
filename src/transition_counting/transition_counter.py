import numpy as np
from transition_counting.frame_analyzer import FrameAnalyzer


class TransitionCounter:
    def count_transitions(self, file_data: dict, step: int, starting_step: int) -> np.ndarray:
        """
        Counts transitions.
        :return: return 2x2x2x2 matrix that translates to:
            [Other_previous_state][Other_current_state][Main_previous_state][Main_current_state]
        """
        result = np.zeros((2, 2, 2, 2))

        previous_frame = None

        #take only nth item at list
        for frame in file_data[starting_step::step]:
            if previous_frame is None:
                previous_frame = frame
                continue

            frame_analyzer = FrameAnalyzer()
            this_frame_result = frame_analyzer.process_frame(previous_frame, frame)
            result = result + this_frame_result

        return result
