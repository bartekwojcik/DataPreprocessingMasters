from data_const import JointConstants as ConstJoint


class GazeProcessor():

    def __init__(self):
        self.gaze_states_in_list = [
            ConstJoint.MOUTH,
            ConstJoint.LEFT_EYE,
            ConstJoint.RIGHT_EYE,
        ]
        self.gaze_states_out_list = [ConstJoint.OUT]

    def get_state_id(self, state):
        """
        Return 1 if looks at the face, 0 if not
        :param state:
        :return:
        """
        if state in self.gaze_states_in_list:
            return 1
        else:
            return 0

    def decode_matrix(self, matrix, other_previous_state: str, other_current_state, main_previous_state,
                      main_current_state):
        """
         return values from matrix and given states[Other_previous_state][Other_current_state][Main_previous_state][Main_current_state]
        :return:
        """

        other_previous = self.get_state_id(other_previous_state)
        other_current = self.get_state_id(other_current_state)
        main_previous = self.get_state_id(main_previous_state)
        main_current = self.get_state_id(main_current_state)

        values =  matrix[other_previous][other_current][
            main_previous
        ][main_current]

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
