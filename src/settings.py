import os


class Settings:
    def __init__(
        self,
        MAX_CONTINUOUS_TIME_SEC: float,
        DISCOUNT_FACTOR: float,
        POLICY_THETA: float,
        IRL_SOLVER_EPSILON: float,
        GLOBAL_PREFIX_FOR_FILE_NAMES: str = "",
    ):
        # time
        self.MAX_CONTINUOUS_TIME_SEC = MAX_CONTINUOUS_TIME_SEC
        # each step in data is like that
        self.CONTINUOUS_TIME_STEP_SEC = 0.04
        self.TIME_SIZE = int(
            self.MAX_CONTINUOUS_TIME_SEC / self.CONTINUOUS_TIME_STEP_SEC
        )

        self.GLOBAL_PREFIX_FOR_FILE_NAMES = GLOBAL_PREFIX_FOR_FILE_NAMES

        self.DISCOUNT_FACTOR = DISCOUNT_FACTOR
        self.POLICY_THETA = POLICY_THETA
        self.IRL_SOLVER_EPSILON = IRL_SOLVER_EPSILON

        self.TRANSITION_FRAME_STEP = 1

    SOURCE_ROOT = os.path.abspath(os.path.dirname(__file__))  # type: str
    PROJECT_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )  # type: str
    MY_DATA_FOLDER_PATH = os.path.join(PROJECT_ROOT, "my-data")
    CLUSTER_DATA_FOLDER_PATH = os.path.join(
        PROJECT_ROOT, "original-data", "ClusterData", "ExpandedRect1.5"
    )  # type: str
    JOINT_DATA_FOLDER_PATH = os.path.join(
        PROJECT_ROOT, "original-data", "JointData"
    )  # type: str
    USABLE_CONVERSATIONS_FILE_PATH = os.path.join(
        PROJECT_ROOT, "original-data", "ClusterData", "UsableConversations.json"
    )  # type: str
    HUMAN_READABLE_FOLDER_PATH = os.path.join(
        PROJECT_ROOT, MY_DATA_FOLDER_PATH, "human-readable-conversations"
    )  # type: str

    READABLE_METADATA_FILE_PATH = os.path.join(
        MY_DATA_FOLDER_PATH, "human-readable-conversation-metadata.json"
    )
    COMPARISON_PLOTS_FOLDER_PATH = os.path.join(
        MY_DATA_FOLDER_PATH, "comparisons_plots"
    )
    TRANSITION_RESULTS_FOLDER_PATH = os.path.join(
        MY_DATA_FOLDER_PATH, "transition_results"
    )
    HISTOGRAMS_FOLDER_PATH = os.path.join(MY_DATA_FOLDER_PATH, "histograms")
