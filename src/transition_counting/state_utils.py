from typing import Tuple

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
    def get_gaze_id(state:str)->int:
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
    def get_talk_id(state:str)->int:
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
    def gaze_id_to_string(cls, id: int)->str:
        return JointConstants.LEFT_EYE if id == MdpConsts.LOOK else JointConstants.OUT

    @classmethod
    def talk_id_to_string(cls, id: int)->str:
        return JointConstants.TALKING if id == MdpConsts.TALK else JointConstants.QUIET

    @classmethod
    def state_vector_to_human_string(cls, vector: Tuple[int,int,int,int,int]):
        h_g = vector[0]
        h_t = vector[1]
        l_g = vector[2]
        l_t = vector[3]

        return f"state: high person is {cls.state_to_simple_string(h_g,h_t)} " \
               f"and low person is {cls.state_to_simple_string(l_g,l_t)}" \


    @classmethod
    def state_to_simple_string(cls, gaze_state:int, talk_state:int):
        """
        produces simple string like "looking at and silent
        :param gaze_state:
        :param talk_state:
        :return:
        """
        gaze_dict = {
            0: "looking away",
            1: "looking at"
        }

        talk_dict = {
            0: "silent",
            1: "talking"
        }

        return f"{gaze_dict[gaze_state]} and {talk_dict[talk_state]}"

