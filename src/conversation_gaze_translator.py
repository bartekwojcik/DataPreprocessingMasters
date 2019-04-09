import json
from data_const import ClusterConstants as ConstCluster, JointConstants as ConstJoint
from utils import *

"""
Converts conversation (per frame) file from gaze x and y coordinates to gaze "mouth" or gaze "out"
"""


class ConversationGazeTranslator:
    """
    Converts conversation (per frame) file from gaze x and y coordinates to gaze "mouth" or gaze "out"
    """

    def __init__(
        self,
        joint_data: dict,
        clustering_data: dict,
        main_person: str,
        num_clusters: int,
    ):
        self.__main_person = main_person
        self.__num_clusters = num_clusters
        self.__clustering_data = clustering_data
        self.__joint_data = joint_data
        self.__res_info = self.__extract_resolution_info(joint_data)
        self.__centorids = self.__extract_centroids(self.__clustering_data, main_person, num_clusters)

    def __extract_resolution_info(self, joint_data: dict) -> dict:
        """
        Return dictionary with extra info fom joint_data 
        :param joint_data: 
        :return: 
        """
        extraInfo = joint_data["extraInfo"]
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

    def convert_to_readable(self) -> dict:
        """
        Converts join_data and cluster_data to more human-readable format
        :return: dict
        """
        pass

    def __extract_centroids(
        self, __clustering_data: dict, person: str, cluster: int
    ) -> np.ndarray:
        clusters_info = __clustering_data[person][ConstCluster.FACE_KMEANS][cluster-1][ConstCluster.CENTROIDS]
        result_vectors = [Utils.dict_to_vec(key_val) for key_val in clusters_info]
        result = np.array(result_vectors)
        return result


