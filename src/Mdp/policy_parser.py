from data_const import JointConstants as consts
from typing import List

from human_read_creator.utils import Utils
from Mdp.mdp_utils import MdpUtils
from transition_counting.gaze_utils import GazeUtils
from data_const import ReadableConvMetadataConstants as readConsts


class PolicyParser:
    """
    Parses data from human_readable files to policy as a dict of {state:action}
    """

    @staticmethod
    def parse_data_to_policy(data: dict, metadata:dict) -> List[dict]:
        """
        :param data: original data from human_readable_file
        :param metadata: metadata from my-data/human-readable-conversation-metadata.json
        """

        policy = PolicyParser.__process_data(data, metadata)
        return policy

    @staticmethod
    def __process_data(data: dict, metadata:dict) -> List[dict]:
        """
        Parses data from human_readable_file as a list of {state:action}
        :return: Policy
        """
        policy = []

        previous_frame = {}
        for frame in data:
            policy_entry = PolicyParser.__process_frame(frame, previous_frame, metadata[readConsts.AT_HIGH])
            previous_frame = frame
            if not policy_entry:
                continue
            policy.append(policy_entry)

        return policy

    @staticmethod
    def __process_frame(current_frame: dict, previous_frame: dict, at_high_person:int) -> dict:
        """
        return a dict of {state:action}
        :param current_frame: dict of data
        :param previous_frame: dict of data
        :param at_high_person: person1 or person2 that is at_high at this conversation
        :return: 
        """
        if not previous_frame:
            return {}

        main_person = at_high_person
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

        action = MdpUtils.get_action(first_state, end_state)

        return {first_state: action}
