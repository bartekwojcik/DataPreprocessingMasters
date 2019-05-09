from data_const import JointConstants as consts
from typing import List

from human_read_creator.utils import Utils
from inverse_reinforcement_learning.mdp_utils import MdpUtils
from mdp_const import create_action_name
from transition_counting.gaze_utils import GazeUtils
from data_const import UsableConversationConstants as usableConsts


class PolicyParser:
    """
    Parses data from human_readable files to policy as a dict of {state:action}
    """
    @staticmethod
    def parse_data_to_policy(data: dict) -> List[dict]:
        """
        :param data: original data from human_readable_file
        """

        policy = PolicyParser.__process_data(data)
        return policy
    @staticmethod
    def __process_data(data: dict) -> List[dict]:
        """
        Parses data from human_readable_file as a list of {state:action}
        :return: Policy
        """
        policy = []

        previous_frame = {}
        for frame in data:
            policy_entry = PolicyParser.__process_frame(frame, previous_frame)
            previous_frame = frame
            if not policy_entry:
                continue
            policy.append(policy_entry)

        return policy

    @staticmethod
    def __process_frame(current_frame: dict, previous_frame: dict) -> dict:
        """
        return a dict of {state:action}
        :param current_frame: 
        :param previous_frame: 
        :return: 
        """
        if not previous_frame:
            return {}

        main_person = current_frame[usableConsts.MAIN]
        other_peron = Utils.people_dict[main_person]

        current_person1 = current_frame[main_person]
        current_person1_gaze = GazeUtils.get_gaze_id(current_person1[consts.GAZE])

        current_person2 = current_frame[other_peron]
        current_person2_gaze = GazeUtils.get_gaze_id(current_person2[consts.GAZE])

        previous_person1 = previous_frame[main_person]
        previous_person1_gaze = GazeUtils.get_gaze_id(previous_person1[consts.GAZE])

        previous_person2 = previous_frame[other_peron]
        previous_person2_gaze = GazeUtils.get_gaze_id(previous_person2[consts.GAZE])

        end_state = MdpUtils.get_state(current_person1_gaze, current_person2_gaze)
        first_state = MdpUtils.get_state(previous_person1_gaze, previous_person2_gaze)

        action = create_action_name(first_state, end_state)
        return {first_state: action}
