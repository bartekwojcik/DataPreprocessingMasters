from data_const import JointConstants as ConstJoint

class GazeUtils:

    gaze_states_in_list = [
        ConstJoint.MOUTH,
        ConstJoint.LEFT_EYE,
        ConstJoint.RIGHT_EYE,
    ]
    gaze_states_out_list = [ConstJoint.OUT]


    @staticmethod
    def get_gaze_id(state):
        """
        Return 1 if looks at the face, 0 if not
        :param state:
        :return:
        """
        if state in GazeUtils.gaze_states_in_list:
            return 1
        else:
            return 0