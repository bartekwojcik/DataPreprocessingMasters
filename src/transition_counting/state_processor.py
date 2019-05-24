from data_const import JointConstants as ConstJoint
from transition_counting.state_utils import StateUtils


class StateProcessor:

    def get_state_id(self, state) -> int:
        """
        Return 1 if looks at the face, 0 if not
        :param state:
        :return:
        """
        return StateUtils.get_gaze_id(state)

    def get_talk_id(self, state) -> int:
        """
        Return 1 if looks at the face, 0 if not
        :param state:
        :return:
        """
        return StateUtils.get_talk_id(state)


    def increment_matrix(
        self,
        matrix,
            previous_high_gaze_state: str, current_high_gaze_state : str,
            previous_high_talk_state : str, current_high_talk_state : str,
            previous_low_gaze_state : str, current_low_gaze_state: str,
            previous_low_talk_state: str, current_low_talk_state: str,
        increment_value: int,
    ):
        """
         increment values in matrix given GAZE
         :return: return 2x2x2x2x2x2x2x2 matrix that translates to:
             previous_high_gaze_state, current_high_gaze_state,
            previous_high_talk_state, current_high_talk_state,
            previous_low_gaze_state, current_low_gaze_state,
            previous_low_talk_state, current_low_talk_state,
         by increment_value
        """

        low_previous_gaze = self.get_state_id(previous_low_gaze_state)
        low_current_gaze = self.get_state_id(current_low_gaze_state)
        high_previous_gaze = self.get_state_id(previous_high_gaze_state)
        high_current_gaze = self.get_state_id(current_high_gaze_state)

        low_previous_talk = self.get_talk_id(previous_low_talk_state)
        low_current_talk = self.get_talk_id(current_low_talk_state)
        high_previous_talk = self.get_talk_id(previous_high_talk_state)
        high_current_talk = self.get_talk_id(current_high_talk_state)

        matrix[
            high_previous_gaze][high_previous_talk][
            low_previous_gaze][low_previous_talk][
            high_current_gaze][high_current_talk][
            low_current_gaze][low_current_talk
        ] += increment_value

        return matrix
