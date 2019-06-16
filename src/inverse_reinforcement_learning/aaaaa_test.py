from Mdp.at_high_model_components.at_high_policy_player import HighPolicyPlayer
from Mdp.mdp_utils import MdpUtils
import numpy as np
import json

from inverse_reinforcement_learning.conversation_comperar import ConversationComparer

from settings import Settings

file_path = "C:\\Users\\kicjo\\Documents\\PythonProjects\\DataPreprocessing-Masters\\my-data\\comparisons_plots\\frame_1_time_200 - Copy\\_human_readable_conversation_16.json_policies.npy"
policy = np.load(file_path)[12]

settings = Settings(MAX_CONTINUOUS_TIME_SEC=8.0,
                    DISCOUNT_FACTOR=0.999999,
                    POLICY_THETA=0.01,
                    IRL_SOLVER_EPSILON=0.05,
                    Q_ITERATIONS=100,
                    Q_ALPHA=0.4,
                    Q_EPSILON=0.2)

model = MdpUtils.get_at_high_mdp_model(settings)



METADATA_PATH = settings.READABLE_METADATA_FILE_PATH

with open(METADATA_PATH, "r") as metadata_file:
    metadata_json = json.loads(metadata_file.read())

    metadata = metadata_json["human_readable_conversation_16.json"]

    player = HighPolicyPlayer(metadata,model,0.05)

    conversation = player.play_policy(policy,1200)

    comparer = ConversationComparer()
    comparer.compare_and_save_plots("16_with_made_up",conversation,conversation,1,metadata,True,model.Ca.shape,settings)




