import os

SOURCE_ROOT = os.path.abspath(os.path.dirname(__file__))  # type: str
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # type: str
CLUSTER_DATA_FOLDER_PATH = os.path.join(PROJECT_ROOT, "original-data", "ClusterData", "ExpandedRect1.5")  # type: str
JOINT_DATA_FOLDER_PATH = os.path.join(PROJECT_ROOT, "original-data", "JointData")  # type: str
USABLE_CONVERSATIONS_FILE_PATH = os.path.join(PROJECT_ROOT, "original-data", "ClusterData",
                                              "UsableConversations.json")  # type: str
HUMAN_READABLE_FOLDER_PATH = os.path.join(PROJECT_ROOT, "my-data", "human-readable-conversations")  # type: str
MY_DATA_FOLDER_PATH = os.path.join(PROJECT_ROOT, "my-data")

