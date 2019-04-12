from typing import List

import numpy as np
from data_const import JointConstants


class Utils:
    @staticmethod
    def dict_to_vec(json: dict) -> np.ndarray:
        """
        Return numpy vector 2d given dictionary that has attribute "x" and "y"
        :param json:
        :return:
        """
        return np.array([json[JointConstants.X], json[JointConstants.Y]]).astype(float)

    @staticmethod
    def vec_to_dict(vector: np.ndarray) -> dict:
        """
        Return Dict with attributes "x" and "y"
        :param vector:
        :return:
        """
        return {JointConstants.X: vector[0], JointConstants.Y: vector[1]}

    @staticmethod
    def any_string_in_list(list_of_strings: List[str], searched_string: str):
        """
        Checks if any string in the list is the same as searched_string
        :param list_of_strings:
        :param searched_string:
        :return:
        """
        return any(item == searched_string for item in list_of_strings)

    # k-means base function taken from http://flothesof.github.io/k-means-numpy.html
    @staticmethod
    def closest_centroid(points, centroids):
        """returns an array containing the index to the nearest centroid for each point"""
        distances = np.sqrt(((points - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        return np.argmin(distances, axis=0)
