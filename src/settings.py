import os


class Settings:
    def __init__(
        self,
        MAX_CONTINUOUS_TIME_SEC: float = 10,
        DISCOUNT_FACTOR: float = 0.9999,
        IRL_SOLVER_EPSILON: float = 0.5,
        GLOBAL_PREFIX_FOR_FILE_NAMES: str = "",
        Q_ITERATIONS = 700,
        Q_ALPHA = 0.5,
        Q_EPSILON = 0.05,
        CONTINUOUS_TIME_STEP_SEC = 0.04
    ):

        self.Q_EPSILON = Q_EPSILON
        self.Q_ALPHA = Q_ALPHA
        self.Q_ITERATIONS = Q_ITERATIONS
        # time
        self.MAX_CONTINUOUS_TIME_SEC = MAX_CONTINUOUS_TIME_SEC
        # each step in data is like that
        self.CONTINUOUS_TIME_STEP_SEC = CONTINUOUS_TIME_STEP_SEC
        self.TIME_SIZE = int(
            self.MAX_CONTINUOUS_TIME_SEC / self.CONTINUOUS_TIME_STEP_SEC
        )

        self.GLOBAL_PREFIX_FOR_FILE_NAMES = GLOBAL_PREFIX_FOR_FILE_NAMES

        self.DISCOUNT_FACTOR = DISCOUNT_FACTOR
        self.IRL_SOLVER_EPSILON = IRL_SOLVER_EPSILON

        self.TRANSITION_FRAME_STEP = 1

        self.SOURCE_ROOT = os.path.abspath(os.path.dirname(__file__))  # type: str
        self.PROJECT_ROOT = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )  # type: str
        self.MY_DATA_FOLDER_PATH = os.path.join(self.PROJECT_ROOT, "my-data")
        self.CLUSTER_DATA_FOLDER_PATH = os.path.join(
            self.PROJECT_ROOT, "original-data", "ClusterData", "ExpandedRect1.5"
        )  # type: str
        self.JOINT_DATA_FOLDER_PATH = os.path.join(
            self.PROJECT_ROOT, "original-data", "JointData"
        )  # type: str
        self.USABLE_CONVERSATIONS_FILE_PATH = os.path.join(
            self.PROJECT_ROOT, "original-data", "ClusterData", "UsableConversations.json"
        )  # type: str
        self.HUMAN_READABLE_FOLDER_PATH = os.path.join(
            self.PROJECT_ROOT, self.MY_DATA_FOLDER_PATH, "human-readable-conversations"
        )  # type: str

        self.READABLE_METADATA_FILE_PATH = os.path.join(
            self.MY_DATA_FOLDER_PATH, "human-readable-conversation-metadata.json"
        )
        self.COMPARISON_PLOTS_FOLDER_PATH = os.path.join(
            self.MY_DATA_FOLDER_PATH, "comparisons_plots"
        )
        self.TRANSITION_RESULTS_FOLDER_PATH = os.path.join(
            self.MY_DATA_FOLDER_PATH, "transition_results"
        )
        self.HISTOGRAMS_FOLDER_PATH = os.path.join(self.MY_DATA_FOLDER_PATH, "histograms")


