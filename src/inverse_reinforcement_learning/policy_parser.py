from data_const import JointConstants as consts
from typing import List

from inverse_reinforcement_learning.mdp_utils import MdpUtils
from mdp_const import create_action_name
from transition_counting.gaze_utils import GazeUtils


class PolicyParser:
    """
    Parses data from human_readable files to policy as a dict of {state:action}
    """

    def parse_data_to_policy(self, data: dict) -> List[dict]:
        """
        :param data: original data from human_readable_file
        """

        policy = self.__process_data(data)
        return policy

    def __process_data(self, data: dict) -> List[dict]:
        """
        Parses data from human_readable_file as a list of {state:action}
        :return: Policy
        """
        policy = []

        previous_frame = {}
        for frame in data:
            policy_entry = self.__process_frame(frame, previous_frame)
            previous_frame = frame
            if not policy_entry:
                continue
            policy.append(policy_entry)

        return policy

    def __process_frame(self, current_frame: dict, previous_frame: dict) -> dict:
        """
        return a dict of {state:action}
        :param current_frame: 
        :param previous_frame: 
        :return: 
        """
        if not previous_frame:
            return {}

        current_person1 = current_frame[consts.PERSON_1]
        current_person1_gaze = GazeUtils.get_gaze_id(current_person1[consts.GAZE])

        current_person2 = current_frame[consts.PERSON_2]
        current_person2_gaze = GazeUtils.get_gaze_id(current_person2[consts.GAZE])

        previous_person1 = previous_frame[consts.PERSON_1]
        previous_person1_gaze = GazeUtils.get_gaze_id(previous_person1[consts.GAZE])

        previous_person2 = previous_frame[consts.PERSON_2]
        previous_person2_gaze = GazeUtils.get_gaze_id(previous_person2[consts.GAZE])

        first_state = MdpUtils.get_state(current_person1_gaze, current_person2_gaze)
        end_state = MdpUtils.get_state(previous_person1_gaze, previous_person2_gaze)

        action = create_action_name(first_state, end_state)
        return {first_state: action}
