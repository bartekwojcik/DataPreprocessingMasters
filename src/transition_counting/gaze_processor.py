from data_const import JointConstants as ConstJoint
from transition_counting.gaze_utils import GazeUtils


class GazeProcessor():

    def __init__(self):

        self.gaze_states_in_list = GazeUtils.gaze_states_in_list
        self.gaze_states_out_list = GazeUtils.gaze_states_out_list

    def get_state_id(self, state)->int:
        """
        Return 1 if looks at the face, 0 if not
        :param state:
        :return:
        """
        return GazeUtils.get_gaze_id(state)

    def decode_matrix(self, matrix, other_previous_state: int, other_current_state: int, main_previous_state: int,
                      main_current_state: int):
        """
        return values from matrix and given states[Other_previous_state][Other_current_state][Main_previous_state][Main_current_state]
        :param matrix: matrix to decode
        :param other_previous_state: 0 or 1 (Zero is out, One is in)
        :param other_current_state: 0 or 1 (Zero is out, One is in)
        :param main_previous_state: 0 or 1 (Zero is out, One is in)
        :param main_current_state: 0 or 1 (Zero is out, One is in)
        :return:
        """

        values =  matrix[other_previous_state][other_current_state][
            main_previous_state
        ][main_current_state]

        return values

    def increment_matrix(self, matrix, other_previous_state: str, other_current_state, main_previous_state,
                      main_current_state, increment_value):
        """
         increment values in matrix given states[Other_previous_state][Other_current_state][Main_previous_state][Main_current_state]
         by increment_value
        :return:
        """

        other_previous = self.get_state_id(other_previous_state)
        other_current = self.get_state_id(other_current_state)
        main_previous = self.get_state_id(main_previous_state)
        main_current = self.get_state_id(main_current_state)

        matrix[other_previous][other_current][
            main_previous
        ][main_current] += increment_value

        return matrix
