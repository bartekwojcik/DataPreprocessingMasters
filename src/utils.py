import numpy as np
from data_const import JointConstants


class Utils:
    @staticmethod
    def dict_to_vec(json: dict) -> np.ndarray:
        return np.array([json[JointConstants.X], json[JointConstants.Y]]).astype(float)

    @staticmethod
    def vec_to_dict(vector: np.ndarray) -> dict:
        return {JointConstants.X: vector[0], JointConstants.Y: vector[1]}
