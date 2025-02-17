from Mdp.at_high_model_components.at_high_model import AtHighMdpModel
from Mdp.at_high_model_components.at_high_policy_player import HighPolicyPlayer
from inverse_reinforcement_learning.feature_expectations_extractor import (
    FeatureExpectationExtractor,
)
from mdp_const import MdpConsts
from Mdp.mdp_utils import MdpUtils
import numpy as np
import json
import scipy.special
from scipy import stats
from settings import Settings
from Mdp.transition_counting_translator import TransitionCountingTranslator
import os
import re
import math
from transition_counting.transition_counter import TransitionCounter
import matplotlib.pyplot as plt
import histograms_creator.main


def plot_kullback_leibler(
    WORK_FOLDER_PATH,
    HISTOGRAMS_FOLDER_PATH,
    TIME_SIZE,
    FRAME_STEP,
    READABLE_METADATA_FILE_PATH,
    HUMAN_READABLE_FOLDER_PATH,
):
    ca_shape = (2, 2, 2, 2, 2, 2, 2, 2, TIME_SIZE)
    METADATA_PATH = READABLE_METADATA_FILE_PATH
    T_FILE_GROUP_NAME = "FILE_NUMBER"
    T_FILE_REGEX = (
        "(human_readable_conversation_)(?P<FILE_NUMBER>[0-9]+)(\.json_T_values\.npy)"
    )

    p = re.compile(T_FILE_REGEX)
    states = MdpConsts.GET_TALK_AND_LOOK_STATES_WITH_TIME(TIME_SIZE)

    with open(METADATA_PATH, "r") as metadata_file:
        metadata_json = json.loads(metadata_file.read())

        counter = TransitionCounter()
        for saved_numpy_file_name in os.listdir(WORK_FOLDER_PATH):

            reg_res = p.match(saved_numpy_file_name)
            if reg_res:
                conv_number = reg_res.group(T_FILE_GROUP_NAME)

                conversation_file_name = (
                    f"human_readable_conversation_{conv_number}.json"
                )
                conversation_file_path = os.path.join(
                    HUMAN_READABLE_FOLDER_PATH, conversation_file_name
                )

                policies_file_name = saved_numpy_file_name.replace(
                    ".json_T_values.npy", ".json_policies.npy"
                )
                policies_file_path = os.path.join(WORK_FOLDER_PATH, policies_file_name)

                with open(conversation_file_path, "r") as conversation_file_string:
                    conv_json = json.load(conversation_file_string)

                    metadata = metadata_json[conversation_file_name]
                    feature_expectation_extractor = FeatureExpectationExtractor(
                        states, metadata, 0.9999999
                    )

                    original_ca = counter.count_transitions(
                        conv_json, FRAME_STEP, 0, metadata, ca_shape, TIME_SIZE
                    )
                    policies = np.load(policies_file_path)
                    t_values = np.load(
                        os.path.join(WORK_FOLDER_PATH, saved_numpy_file_name)
                    )

                    current_model = AtHighMdpModel(original_ca, TIME_SIZE)

                    this_file_kls = []
                    this_file_cas = []

                    for i_pol, policy in enumerate(policies):
                        player = HighPolicyPlayer(metadata, current_model)
                        created_conversation = player.play_policy(
                            policy, len(conv_json)
                        )
                        created_ca = counter.count_transitions(
                            created_conversation,
                            FRAME_STEP,
                            0,
                            metadata,
                            ca_shape,
                            TIME_SIZE,
                        )

                        created_ca_no_zeroes = np.where(created_ca == 0, 1, created_ca)
                        kl = scipy.special.kl_div(original_ca, created_ca_no_zeroes)

                        kl = np.where(kl == math.inf, 0, kl)
                        sum_kl = np.sum(kl)

                        this_file_kls.append(sum_kl)
                        this_file_cas.append(created_ca)

                    # kls[conversation_file_name] = this_file_kls

                    top3 = sorted(this_file_kls)[:3]

                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    print(
                        f"CONVERSATION {conv_number} ######################################"
                    )
                    for x, y in enumerate(this_file_kls):
                        this_x_t_val = np.round(t_values[x], decimals=1)
                        this_x_kl = np.round(y, decimals=1)
                        ax.annotate(
                            f"{x}, kl={this_x_kl}, t={this_x_t_val}",
                            xy=(x, y),
                            textcoords="data",
                        )
                        print(f"{x}, kl={this_x_kl}, t={this_x_t_val}")

                        if y in top3:
                            x_ca = this_file_cas[x]
                            histograms_folder = os.path.join(
                                HISTOGRAMS_FOLDER_PATH, f"kl_{np.round(this_x_kl,2)}"
                            )
                            histograms_creator.main.plot_histograms(
                                conversation_file_name,
                                x_ca,
                                TIME_SIZE,
                                histograms_folder,
                            )

                    plt.plot(this_file_kls)
                    plt.yscale("log")

                    fig.suptitle(
                        f"kl_div for conversation {conv_number}",
                        fontsize=14,
                        fontweight="bold",
                    )

                    plt.ylabel("kullback leibler divergence")
                    plt.xlabel("policy number")

                    file_path = os.path.join(WORK_FOLDER_PATH, f"kl_{conv_number}")
                    fig.set_size_inches((18, 9), forward=False)
                    plt.savefig(file_path, quality=70, dpi=400)
                    plt.close(fig)



"""
this script: 
- point it to the folder with all the .npy files
then:

1) it will find for each conversation it's .npy file with policies and t_values
2) it will find count_transition for given conversation
3) calculate count_transition for each policy from conversation's policies.npy file
4) will produce kl_div value for these two and store it
5) produces plot telling you which policy produces  
"""

if __name__ == "__main__":
    #these two should be adjusted
    WORK_FOLDER_PATH = "C:\\Users\\kicjo\\Documents\\PythonProjects\\DataPreprocessing-Masters\\my-data\\human_readable_conversation_11.json\\policies"
    HISTOGRAMS_FOLDER_PATH = os.path.join(WORK_FOLDER_PATH, "histograms") #place where results will be stored

    settings = Settings()
    TIME_SIZE = settings.TIME_SIZE
    FRAME_STEP = settings.TRANSITION_FRAME_STEP

    plot_kullback_leibler(
        WORK_FOLDER_PATH,
        HISTOGRAMS_FOLDER_PATH,
        TIME_SIZE,
        FRAME_STEP,
        settings.READABLE_METADATA_FILE_PATH,
        settings.HUMAN_READABLE_FOLDER_PATH,
    )
