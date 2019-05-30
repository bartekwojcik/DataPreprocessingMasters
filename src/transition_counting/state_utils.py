from data_const import JointConstants as ConstJoint
from mdp_const import MdpConsts
from data_const import JointConstants

class StateUtils:

    gaze_states_in_list = [
        ConstJoint.MOUTH,
        ConstJoint.LEFT_EYE,
        ConstJoint.RIGHT_EYE,
    ]
    gaze_states_out_list = [ConstJoint.OUT]

    talk_state = ConstJoint.TALKING
    quiet_state = ConstJoint.QUIET

    @staticmethod
    def get_gaze_id(state:str):
        """
        Return 1 if looks at the face, 0 if not
        :param state:
        :return:
        """
        if state in StateUtils.gaze_states_in_list:
            return 1
        else:
            return 0

    @staticmethod
    def get_talk_id(state:str):
        """
        return 1 if "talking" and 0 if "quiet"
        :param state:
        :return:
        """

        if state == StateUtils.talk_state:
            return 1
        else:
            return 0


    @classmethod
    def gaze_id_to_string(cls, id: int):
        return JointConstants.LEFT_EYE if id == MdpConsts.LOOK else JointConstants.OUT

    @classmethod
    def talk_id_to_string(cls, id: int):
        return JointConstants.TALKING if id == MdpConsts.TALK else JointConstants.QUIET

