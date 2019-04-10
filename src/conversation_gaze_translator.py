import json
import math
from typing import Tuple
from cluster_matcher import ClusterMatcher
from data_const import ClusterConstants as ConstCluster, JointConstants as ConstJoint
from utils import *
from face import Face

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
        other_person: str,
        num_clusters: int,
    ):
        self.__main_person = main_person
        self.__other_person = other_person
        self.__num_clusters = num_clusters
        self.__clustering_data = clustering_data
        self.__joint_data = joint_data
        self.__res_info = self.__extract_resolution_info(joint_data)
        self.__centroids_main = self.__extract_centroids(
            self.__clustering_data, self.__main_person, self.__num_clusters
        )
        self.__centroids_other = self.__extract_centroids(
            self.__clustering_data, self.__main_person, self.__num_clusters
        )

        self.__cluster_matcher = ClusterMatcher()

    def __extract_centroids(
        self, clustering_data: dict, person: str, cluster: int
    ) -> np.ndarray:
        """
        Extracts centroids coordinations from cluster data
        :param clustering_data: dict
        :param person: person1 or person2
        :param cluster: number of clusters
        :return:
        """
        clusters_info = clustering_data[person][ConstCluster.FACE_KMEANS][cluster - 1][
            ConstCluster.CENTROIDS
        ]
        result_vectors = [Utils.dict_to_vec(key_val) for key_val in clusters_info]
        result = np.array(result_vectors)
        return result

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

    def __construct_faces(
        self, frame: dict, main_extra_ration: float, other_extra_ratio: float
    ) -> Tuple[Face, Face]:
        """
        Constructs coordinates of faces that belong to main and other person of given frame
        :return: Tuple[Face,Face] where first face is Main person, the second one is Other person
        """
        main_person_frame_data = frame[self.__main_person]
        other_person_frame_data = frame[self.__other_person]
        main_face = Face(
            250,
            main_person_frame_data[ConstJoint.LEFT_EYE],
            main_person_frame_data[ConstJoint.RIGHT_EYE],
            main_person_frame_data[ConstJoint.MOUTH],
            main_extra_ration,
        )
        other_face = Face(
            250,
            other_person_frame_data[ConstJoint.LEFT_EYE],
            other_person_frame_data[ConstJoint.RIGHT_EYE],
            other_person_frame_data[ConstJoint.MOUTH],
            other_extra_ratio,
        )

        return main_face, other_face

    def convert_to_readable(self) -> dict:
        """
        Converts join_data and cluster_data to more human-readable format
        :return: dict
        """

        centroids_with_labels_main = self.__cluster_matcher.label_centroids_heuristically(
            self.__centroids_main
        )

        centroids_with_labels_other = self.__cluster_matcher.label_centroids_heuristically(
            self.__centroids_other
        )

        # TODO delete these comments if not required when the method is finished
        # wanted_image_size = np.array([800,600])
        # img_size_main = Utils.dict_to_vec(self.__res_info[self.__main_person])
        # img_size_other = Utils.dict_to_vec(self.__res_info[self.__other_person])
        # invert_y = np.array([1, -1]).astype(float) # Note: this picture is inverted along axis Y
        # img_scale_main = wanted_image_size / img_size_main * invert_y
        # img_scale_other = wanted_image_size / img_size_other * invert_y

        # # width and height are switched because for some reason pictures are inverted along axis Y
        # desired_face_height = self.__clustering_data[self.__main_person][ConstCluster.DESIRED_FACE_WIDTH]

        extra_space_ratio_main = self.__clustering_data[self.__main_person][
            ConstCluster.EXTRA_FACE_SPACE_RATIO
        ]
        extra_space_ratio_other = self.__clustering_data[self.__other_person][
            ConstCluster.EXTRA_FACE_SPACE_RATIO
        ]
        all_frames = self.__joint_data[ConstJoint.DATA]
        previous_frame = None
        for frame in all_frames:
            if frame[ConstJoint.TYPE == ConstJoint.MISSING]:
                previous_frame = None
                continue
            main_face, other_face = self.__construct_faces(
                frame, extra_space_ratio_main, extra_space_ratio_other
            )



        debug = 5
