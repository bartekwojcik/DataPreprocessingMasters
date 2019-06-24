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

from transition_counting.transition_counter import TransitionCounter

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

            kls = {}
            with open(conversation_file_path, "r") as conversation_file_string:
                conv_json =  json.load(conversation_file_string)

                metadata = metadata_json[conversation_file_name]
                feature_expectation_extractor = FeatureExpectationExtractor(
                    states, metadata, 0.9999999
                )

                FRAME_STEP = settings.TRANSITION_FRAME_STEP
                original_ca = counter.count_transitions(conv_json,FRAME_STEP,0,metadata,ca_shape,settings)
                policies = np.load(policies_file_path)
                current_model = AtHighMdpModel(original_ca, settings)

                this_file_kls = {}

                for policy in policies:

                    player = HighPolicyPlayer(metadata,current_model)
                    created_conversation = player.play_policy(policy,len(conv_json))
                    created_ca = counter.count_transitions(created_conversation,FRAME_STEP,0,metadata,ca_shape,settings)

                    this_kl = scipy.special.kl_div(original_ca,created_ca)
                    sum_kl = np.sum(this_kl)



