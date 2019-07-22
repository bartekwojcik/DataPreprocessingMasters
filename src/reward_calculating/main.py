import numpy as np
import os
import re
import json

from typing import List, Dict, Tuple

from Mdp.at_high_model_components.at_high_model import AtHighMdpModel
from Mdp.at_high_model_components.at_high_policy_player import HighPolicyPlayer
from Mdp.mdp_utils import MdpUtils
from mdp_const import MdpConsts
from settings import Settings
from transition_counting.transition_counter import TransitionCounter
from data_const import (
    ReadableConvMetadataConstants as read_consts,
)
from transition_counting.frame_analyzer import FrameAnalyzer



def get_rewards(conversation:List[Dict], rewards:np.ndarray, metadata)->float:

    time = 0
    previous_frame = None
    final_rewards = 0
    for current_frame in conversation:
        if not previous_frame:
            previous_frame = current_frame
            continue

        state_index, time = get_current_state(current_frame, previous_frame,metadata,time)

        final_rewards += rewards[state_index]

        previous_frame = current_frame

    return final_rewards

def get_current_state(current_frame, previous_frame,metadata, previous_time:int) -> Tuple[int,int]:
    """
    Get vector of current state. For instance [0,0,1,0,TIME], [0,0,0,1,TIME]
    :param current_frame:
    :return:
    """
    high_person = metadata[read_consts.AT_HIGH]
    low_person = metadata[read_consts.AT_LOW]

    current_state_vector_without_time = FrameAnalyzer.get_gaze_talk_state_vector_from_frame(
        current_frame, high_person, low_person
    )

    previous_state_vector_without_time = FrameAnalyzer.get_gaze_talk_state_vector_from_frame(
        previous_frame, high_person, low_person
    )

    current_state_vector = current_state_vector_without_time + (previous_time,)

    if previous_state_vector_without_time == current_state_vector_without_time:
        previous_time = +1
    else:
        previous_time = 0

    states = MdpConsts.GET_TALK_AND_LOOK_STATES_WITH_TIME(settings.TIME_SIZE)

    state_index = states.index(current_state_vector)

    return state_index, previous_time








settings = Settings()
"""
POINT TO THE POLICY .npy FILE, POINT TO THE REWARDS NUMPY FILE
CHANGE VARIABLE ___nr___ to the right policy and rewards number
THIS SCRIPT WILL FIND THE CONVERSATION AND calculate 
reward for original file if it was following calculated rewards
and rewards for calculated file
"""

METADATA_PATH = settings.READABLE_METADATA_FILE_PATH
FOLDER_PATH = "C:\\Users\\kicjo\\Documents\\PythonProjects\\DataPreprocessing-Masters\\my-data\\comparisons_plots\\frame_1_QITERS_700_QEPSILON_0.2-new"
policies_file_name = "QITERS_700_QEPSILON_0.2_human_readable_conversation_99.json_policies.npy"
rewards_file_name = "QITERS_700_QEPSILON_0.2_human_readable_conversation_99.json_rewards.npy"

policies = np.load(os.path.join(FOLDER_PATH,policies_file_name))
rewards = np.load(os.path.join(FOLDER_PATH,rewards_file_name))

#take right policy and rewards
nr = 43
policy = policies[nr]
reward = rewards[nr]

T_FILE_GROUP_NAME = "FILE_NUMBER"
T_FILE_REGEX = "(human_readable_conversation_)" \
               "(?P<FILE_NUMBER>[0-9]+)(\.json_policies\.npy)"
p = re.compile(T_FILE_REGEX)

ca_shape = (2,2,2,2,2,2,2,2,settings.TIME_SIZE)

reg_res = p.match(policies_file_name)
if reg_res:
    conv_number = reg_res.group(T_FILE_GROUP_NAME)

    conversation_file_name = f"human_readable_conversation_{conv_number}.json"
    conversation_file_path = os.path.join(settings.HUMAN_READABLE_FOLDER_PATH,
                                          conversation_file_name)

    with open(conversation_file_path, "r") as conversation_file_string,\
            open(METADATA_PATH, "r") as metadata_file:
        metadata_json = json.loads(metadata_file.read())
        metadata = metadata_json[conversation_file_name]

        conv_json = json.load(conversation_file_string)

        FRAME_STEP = settings.TRANSITION_FRAME_STEP
        counter = TransitionCounter()
        original_ca = counter.count_transitions(conv_json, FRAME_STEP, 0, metadata, ca_shape,settings.TIME_SIZE)
        current_model = AtHighMdpModel(original_ca,settings.TIME_SIZE)
        player = HighPolicyPlayer(metadata, current_model)

        PRVIOUS_TIME = 0
        created_conversation = player.play_policy(policy, len(conv_json))
        original_conv_reward = get_rewards(conv_json,reward,metadata)
        calculated_conv_reward = get_rewards(created_conversation,reward,metadata)

        print(f"conv: {conv_number}, policy/rewwards number: {nr}")
        print(f"original: {np.round(original_conv_reward,decimals=2)}")
        print(f"calculated: {np.round(calculated_conv_reward,decimals=2)}")



