from Mdp.at_high_model_components.at_high_model import AtHighMdpModel
from Mdp.at_high_model_components.at_high_policy_player import HighPolicyPlayer
from inverse_reinforcement_learning.feature_expectations_extractor import FeatureExpectationExtractor
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

"""


get original conversation
get policies
"play" policies
get their transition counts
compare their KL values
"""


settings = Settings(
    MAX_CONTINUOUS_TIME_SEC=10.0,
    DISCOUNT_FACTOR=0.999999,
    POLICY_THETA=0.01,
    IRL_SOLVER_EPSILON=0.05,
    Q_ITERATIONS=100,
    Q_ALPHA=0.4,
    Q_EPSILON=0.2,
)

model = MdpUtils.get_at_high_mdp_model(settings)
ca_shape = model.Ca.shape
METADATA_PATH = settings.READABLE_METADATA_FILE_PATH
T_FILE_GROUP_NAME = "FILE_NUMBER"
T_FILE_REGEX = "(QITERS_)\d+(_QEPSILON_)[0-9].[0-9]+(_human_readable_conversation_)" \
               "(?P<FILE_NUMBER>[0-9]+)(\.json_T_values\.npy)"

p = re.compile(T_FILE_REGEX)
states = model.states


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

with open(METADATA_PATH, "r") as metadata_file:
    metadata_json = json.loads(metadata_file.read())

    FOLDER_PATH = "C:\\Users\\kicjo\\Documents\\PythonProjects\\DataPreprocessing-Masters\\my-data\\comparisons_plots\\frame_1_QITERS_1000_QEPSILON_0.05"

    counter = TransitionCounter()
    for saved_numpy_file_name in os.listdir(FOLDER_PATH):

        reg_res = p.match(saved_numpy_file_name)
        if reg_res:
            conv_number = reg_res.group(T_FILE_GROUP_NAME)

            conversation_file_name = f"human_readable_conversation_{conv_number}.json"
            conversation_file_path = os.path.join(settings.HUMAN_READABLE_FOLDER_PATH,
                                                  conversation_file_name)

            policies_file_name = saved_numpy_file_name.replace(".json_T_values.npy",".json_policies.npy")
            policies_file_path = os.path.join(FOLDER_PATH, policies_file_name)

            with open(conversation_file_path, "r") as conversation_file_string:
                conv_json =  json.load(conversation_file_string)

                metadata = metadata_json[conversation_file_name]
                feature_expectation_extractor = FeatureExpectationExtractor(
                    states, metadata, 0.9999999
                )

                FRAME_STEP = settings.TRANSITION_FRAME_STEP
                original_ca = counter.count_transitions(conv_json,FRAME_STEP,0,metadata,ca_shape,settings)
                policies = np.load(policies_file_path)
                t_values = np.load(os.path.join(FOLDER_PATH,saved_numpy_file_name))

                current_model = AtHighMdpModel(original_ca, settings)

                this_file_kls = []

                for i_pol,policy in enumerate(policies):

                    player = HighPolicyPlayer(metadata,current_model)
                    created_conversation = player.play_policy(policy,len(conv_json))
                    created_ca = counter.count_transitions(created_conversation,FRAME_STEP,0,metadata,ca_shape,settings)


                    #i am not sure if this is supposed to be count_transitions
                    # perhaps 2D probabilities per state would be better?
                    kl = scipy.special.kl_div(original_ca, created_ca)

                    #if value is inf, we want to replace it with 0
                    # (i dont know if this is correct though :D)
                    kl = np.where(kl == math.inf, 0, kl)
                    sum_kl = np.sum(kl)

                    this_file_kls.append(sum_kl)

                #kls[conversation_file_name] = this_file_kls

                fig = plt.figure()
                ax = fig.add_subplot(111)
                for x,y in enumerate(this_file_kls):
                    this_x_t_val = np.round(t_values[x],decimals=1)
                    this_x_kl =  np.round(y,decimals=1)
                    ax.annotate(f"{x}, kl={this_x_kl}, t={this_x_t_val}",xy=(x,y),textcoords = 'data')

                plt.plot(this_file_kls)
                plt.yscale('log')

                fig.suptitle(
                    f"kl_div for conversation {conv_number}", fontsize=14, fontweight="bold"
                )

                plt.ylabel("kullback leibler divergence")
                plt.xlabel("policy number")



                file_path = os.path.join(FOLDER_PATH, f"kl_{conv_number}")
                fig.set_size_inches((18, 9), forward=False)
                plt.savefig(file_path, quality=70, dpi=400)
                plt.close(fig)




