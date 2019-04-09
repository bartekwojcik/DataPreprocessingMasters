import json
from data_const import ClusterConstants as ConstCluster, JointConstants as ConstJoint
"""
Converts conversation (per frame) file from gaze x and y coordinates to gaze "mouth" or gaze "out"
"""


class ConversationGazeTranslator(object):
    """
    Converts conversation (per frame) file from gaze x and y coordinates to gaze "mouth" or gaze "out"
    """


    def __init__(self, joint_data: dict, clustering_data: dict):
        self.__clustering_data = clustering_data
        self.__joint_data = joint_data
        self.__res_info = self.__extract_resolution_info(joint_data)

    def __extract_resolution_info(self, join_data: dict) -> dict:
        extraInfo = join_data["extraInfo"]
        result = {}
        result[ConstJoint.PERSON_1] = {
            ConstJoint.FPS: extraInfo[ConstJoint.PERSON_1][ConstJoint.FPS],
            ConstJoint.X: extraInfo[ConstJoint.PERSON_1]["imageSize"][ConstJoint.X],
            ConstJoint.Y: extraInfo[ConstJoint.PERSON_1]["imageSize"][ConstJoint.Y],
        }
        result[ConstJoint.PERSON_2] = {
            ConstJoint.FPS: extraInfo[ConstJoint.PERSON_2][ConstJoint.FPS],
            ConstJoint.X: extraInfo[ConstJoint.PERSON_2]["imageSize"][ConstJoint.X],
            ConstJoint.Y: extraInfo[ConstJoint.PERSON_2]["imageSize"][ConstJoint.Y],
        }
        return result

    def convert_to_readable(self):
        pass