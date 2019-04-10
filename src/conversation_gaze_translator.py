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
        extra_info = joint_data["extraInfo"]  # type: dict
        result = {
            ConstJoint.PERSON_1: {
                ConstJoint.FPS: extra_info[ConstJoint.PERSON_1][ConstJoint.FPS],
                ConstJoint.X: extra_info[ConstJoint.PERSON_1]["imageSize"][
                    ConstJoint.X
                ],
                ConstJoint.Y: extra_info[ConstJoint.PERSON_1]["imageSize"][
                    ConstJoint.Y
                ],
            },
            ConstJoint.PERSON_2: {
                ConstJoint.FPS: extra_info[ConstJoint.PERSON_2][ConstJoint.FPS],
                ConstJoint.X: extra_info[ConstJoint.PERSON_2]["imageSize"][
                    ConstJoint.X
                ],
                ConstJoint.Y: extra_info[ConstJoint.PERSON_2]["imageSize"][
                    ConstJoint.Y
                ],
            },
        }
        return result

    def __construct_faces(
        self, frame: dict, main_extra_ration: float, other_extra_ratio: float
    ) -> Tuple[Face, Face]:
        """
        Constructs coordinates of faces that belong to main and other person of given frame
        :return: Tuple[Face,Face] where first face is Main person, the second one is Other person
        """
        main_person_frame_data = frame[self.__main_person]  # type: dict
        other_person_frame_data = frame[self.__other_person]  # type: dict
        main_face = Face(
            250,
            Utils.dict_to_vec(
                other_person_frame_data[ConstJoint.LEFT_EYE][ConstJoint.CENTRE]
            ),
            Utils.dict_to_vec(
                other_person_frame_data[ConstJoint.RIGHT_EYE][ConstJoint.CENTRE]
            ),
            Utils.dict_to_vec(
                other_person_frame_data[ConstJoint.MOUTH][ConstJoint.CENTRE]
            ),
            main_extra_ration,
        )
        other_face = Face(
            250,
            Utils.dict_to_vec(
                main_person_frame_data[ConstJoint.LEFT_EYE][ConstJoint.CENTRE]
            ),
            Utils.dict_to_vec(
                main_person_frame_data[ConstJoint.RIGHT_EYE][ConstJoint.CENTRE]
            ),
            Utils.dict_to_vec(
                main_person_frame_data[ConstJoint.MOUTH][ConstJoint.CENTRE]
            ),
            other_extra_ratio,
        )

        return main_face, other_face

    def convert_to_readable(self) -> list:
        """
        Converts join_data and cluster_data to more human-readable format
        :return: dict
        """

        centroids_with_labels_main = self.__cluster_matcher.label_centroids_heuristically(
            self.__centroids_main
        )  # type: List[Tuple[str, dict]]
        centroids_with_labels_other = self.__cluster_matcher.label_centroids_heuristically(
            self.__centroids_other
        )  # type: List[Tuple[str, dict]]

        extra_space_ratio_main = self.__clustering_data[self.__main_person][
            ConstCluster.EXTRA_FACE_SPACE_RATIO
        ]  # type: float
        extra_space_ratio_other = self.__clustering_data[self.__other_person][
            ConstCluster.EXTRA_FACE_SPACE_RATIO
        ]  # type: float

        all_frames = self.__joint_data[ConstJoint.DATA]  # type: dict
        new_frames = []  # type: list
        for frame in all_frames:
            if frame[ConstJoint.TYPE] == ConstJoint.MISSING:
                continue

            main_face, other_face = self.__construct_faces(
                frame, extra_space_ratio_main, extra_space_ratio_other
            )

            gaze_main_vector = Utils.dict_to_vec(
                frame[self.__main_person][ConstJoint.GAZE]
            )  # type: np.ndarray
            gaze_other_vector = Utils.dict_to_vec(
                frame[self.__other_person][ConstJoint.GAZE]
            )  # type: np.ndarray

            gaze_main_state = ConstJoint.OUT  # type: str
            gaze_other_state = ConstJoint.OUT  # type: str

            if main_face.is_inside_face(gaze_other_vector):
                gaze_other_state = self.__get_centroid_label(
                    self.__centroids_main,
                    centroids_with_labels_main,
                    gaze_other_vector,
                    main_face,
                )

            # Note: in fracisco code it was otherFace.isInside(gazeOther)
            if other_face.is_inside_face(gaze_main_vector):
                gaze_other_state = self.__get_centroid_label(
                    self.__centroids_other,
                    centroids_with_labels_other,
                    gaze_main_vector,
                    other_face,
                )

            main_talking = (
                "talking" if frame[self.__main_person][ConstJoint.TALKING] else "quiet"
            )
            other_talking = (
                "talking" if frame[self.__other_person][ConstJoint.TALKING] else "quiet"
            )

            new_frame = self.__create_new_frame(
                frame, gaze_main_state, gaze_other_state, main_talking, other_talking
            )

            new_frames.append(new_frame)

        return new_frames

    def __get_centroid_label(self, centroids, centroids_with_labels, gaze_vector, face):
        """
        Retrieves centroid label that is the closest to the gaze point
        :param centroids:
        :param centroids_with_labels:
        :param gaze_vector:
        :return:
        """

        transformed_gaze = face.transform_to_face(gaze_vector)
        centroidId = Utils.closest_centroid(
            np.array([transformed_gaze]), np.array(centroids)
        )[0]
        gaze_state = centroids_with_labels[centroidId][0]
        return gaze_state

    def __create_new_frame(
        self,
        frame: dict,
        gaze_main_state: str,
        gaze_other_state: str,
        main_talking: str,
        other_talking: str,
    ):
        time_end = frame[ConstJoint.TIME_END]  # type: float
        time_start = frame[ConstJoint.TIME_START]  # type: float
        frame_type = frame[ConstJoint.TYPE]

        new_frame = {
            self.__main_person: {
                ConstJoint.GAZE: gaze_main_state,
                ConstJoint.TALKING: main_talking,
            },
            self.__other_person: {
                ConstJoint.GAZE: gaze_other_state,
                ConstJoint.TALKING: other_talking,
            },
            ConstJoint.TIME_END: time_end,
            ConstJoint.TIME_START: time_start,
            ConstJoint.TYPE: frame_type,
        }
        return new_frame
