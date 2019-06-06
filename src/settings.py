import os

SOURCE_ROOT = os.path.abspath(os.path.dirname(__file__))  # type: str
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # type: str
MY_DATA_FOLDER_PATH = os.path.join(PROJECT_ROOT, "my-data")
CLUSTER_DATA_FOLDER_PATH = os.path.join(PROJECT_ROOT, "original-data", "ClusterData", "ExpandedRect1.5")  # type: str
JOINT_DATA_FOLDER_PATH = os.path.join(PROJECT_ROOT, "original-data", "JointData")  # type: str
USABLE_CONVERSATIONS_FILE_PATH = os.path.join(PROJECT_ROOT, "original-data", "ClusterData",
                                              "UsableConversations.json")  # type: str
HUMAN_READABLE_FOLDER_PATH = os.path.join(PROJECT_ROOT, MY_DATA_FOLDER_PATH, "human-readable-conversations")  # type: str

READABLE_METADATA_FILE_PATH = os.path.join(MY_DATA_FOLDER_PATH, "human-readable-conversation-metadata.json")
COMPARISON_PLOTS_FOLDER_PATH = os.path.join(MY_DATA_FOLDER_PATH, "comparisons_plots")
TRANSITION_RESULTS_FOLDER_PATH = os.path.join(MY_DATA_FOLDER_PATH,"transition_results")
HISTOGRAMS_FOLDER_PATH = os.path.join(MY_DATA_FOLDER_PATH,"histograms")


TRANSITION_FRAME_STEP = 1

GLOBAL_PREFIX_FOR_FILE_NAMES = ""

DISCOUNT_FACTOR = 0.99
POLICY_THETA = 0.001
IRL_SOLVER_EPSILON = 0.1