import numpy as np

"""
Class based on Face class by Francisco Mendonca - Heriot-Watt university 
"""


class Face:
    def __init__(self, wantedHeight, leftEye, rightEye, mouth, extraFaceSpaceRatio=1.5):
        self.m_wantedHeight = float(wantedHeight)
        self.m_wantedWidth = (3.0 / 5.0) * self.m_wantedHeight
        self.m_wantedHeightUnit = self.m_wantedHeight / 16.0
        self.m_wantedWidthUnit = self.m_wantedWidth / 10.0
        self.m_leftEye = np.array(leftEye).astype(float)
        self.m_rightEye = np.array(rightEye).astype(float)
        self.m_mouth = np.array(mouth).astype(float)
        self.m_centre = (self.m_leftEye + self.m_rightEye) * 0.5
        self.m_upDir = self.__normalize(self.m_centre - self.m_mouth)
        self.m_rightDir = np.array([-self.m_upDir[1], self.m_upDir[0]])
        self.m_originalHeight = self.__norm(self.m_centre - self.m_mouth) * 16.0 / 5.0
        self.m_originalWidth = self.__norm(self.m_rightEye - self.m_leftEye) * 5.0 / 2.0
        self.m_scaleUp = self.m_wantedHeight / self.m_originalHeight
        self.m_scaleRight = self.m_wantedWidth / self.m_originalWidth
        self.m_extraFaceSpaceRatio = extraFaceSpaceRatio

    def transform_to_face(self, vector: np.ndarray) -> np.ndarray:
        vec = np.array(vector).astype(float)
        vec = vec - self.m_centre
        return np.array(
            [
                vec.dot(self.m_rightDir) * self.m_scaleRight,
                vec.dot(self.m_upDir) * self.m_scaleUp,
            ]
        )

    def is_gaze_inside(self, vec: np.ndarray) -> bool:


        transformed_vec = self.transform_to_face(vec)
        efsr = self.m_extraFaceSpaceRatio
        if (
            abs(transformed_vec[0]) > 0.5 * self.m_wantedWidth * 1.2 * efsr
        ):  # check if outside horizontally includeng space for ears
            return False
        if (
            abs(transformed_vec[1]) > 0.5 * self.m_wantedHeight * efsr
        ):  # check if outside vertically
            return False
        return True

    def transform_to_face_centred(self, vec, scale):
        vec = np.array(vec).astype(float)
        vec = (vec - self.m_centre) * scale
        return vec

    def __normalize(self, vec):
        vec = np.array(vec).astype(float)
        return np.divide(vec, float(self.__norm(vec)))

    def __norm(self, vec):
        vec = np.array(vec).astype(float)
        return float(np.sqrt(vec.dot(vec)))

