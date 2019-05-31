

class TransitionMatrixUpdater:

    def increment_matrix(
        self,
        matrix,
        high_previous_gaze: int,
        high_previous_talk: int,
        low_previous_gaze: int,
        low_previous_talk: int,
        high_current_gaze: int,
        high_current_talk: int,
        low_current_gaze: int,
        low_current_talk: int,
        time: int,
        increment_value: int,
    ):
        """
         increment values in matrix given gaze, talk and time
         :return: return 2x2x2x2x2x2x2x2xTIME matrix that translates to:
             [high_previous_gaze][
            high_previous_talk][
            low_previous_gaze][
            low_previous_talk][
            high_current_gaze][
            high_current_talk][
            low_current_gaze][
            low_current_talk][
            time]
         by increment_value
        """

        matrix[
            high_previous_gaze][
            high_previous_talk][
            low_previous_gaze][
            low_previous_talk][
            high_current_gaze][
            high_current_talk][
            low_current_gaze][
            low_current_talk][
            time,
        ] += increment_value

        return matrix

