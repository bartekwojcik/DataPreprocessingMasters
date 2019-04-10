from typing import List, Tuple
import numpy as np

from utils import Utils
from data_const import ClusterConstants as ConstCluster, JointConstants as ConstJoint
import math


class ClusterMatcher:
    """
        Matches centroids of clusters with labels "mouth", "leftEye" and "rightEye"
    """
    def __are_labels_matched_with_centroids(
        self, list_of_tuples: List[Tuple[str, dict]], string: str
    ) -> bool:
        return Utils.any_string_in_list(
            list(map(lambda item: item[0], list_of_tuples)), string
        )

    def label_centroids_heuristically(self, centroids):
        """
        Labels centroids to "mouth", "rightEye" or "rightEye"
        :return:
        """

        cluster_centroids_labels = [("", {}) for c in centroids]

        centre_point = centroids[0]
        heuristic_centroids = np.array(
            [
                centre_point + [-30, 30],
                centre_point + [30, 30],
                centre_point + [0, -48.125],
            ]
        )
        heuristic_centroid_labels = [
            ConstJoint.LEFT_EYE,
            ConstJoint.RIGHT_EYE,
            ConstJoint.MOUTH,
        ]
        labeled = [False for c in centroids]
        used_label = [False for c in heuristic_centroids]
        while self.__are_labels_matched_with_centroids(cluster_centroids_labels, ""):
            min_dist_square = math.inf
            min_centroid = 0
            min_cluster = 0
            current_cluster = {}
            for i, c in enumerate(centroids):
                if labeled[i]:
                    continue
                for j, cl in enumerate(heuristic_centroids):
                    if used_label[j]:
                        continue
                    diff = c - cl
                    dist_square = diff.dot(diff)

                    if dist_square < min_dist_square:
                        min_centroid = i
                        current_cluster = c
                        min_cluster = j
                        min_dist_square = dist_square

            cluster_centroids_labels[min_centroid] = (
                heuristic_centroid_labels[min_cluster],
                current_cluster,
            )
            labeled[min_centroid] = True
            used_label[min_cluster] = True

        return cluster_centroids_labels
