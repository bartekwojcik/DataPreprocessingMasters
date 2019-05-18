import numpy as np


class TransitionCountingTranslator:
    def __init__(self, count_matrix: np.ndarray):
        self.counting_matrix = np.array(count_matrix)
        assert self.counting_matrix.shape == (
            2,
            2,
            2,
            2,
        ), "it is not (2,2,2,2) matrix, is it coming from transition_counting_results.npy?"

    def transform_to_4x4_count_matrix(self)->np.ndarray:
        """
        translates matrix 2x2x2x2 matrix of [Low_previous_state][Low_current_state][High_previous_state][High_current_state]
        to 4x4 matrix of actions counting as stated in .xlsx file

        None to None
        None to A at B (High at Low)
        None to B at A (Low at Low)
        None to Mutual

        A at B to None
        A at B to A at B (High at Low) (High at Low)
        A at B to B at A (High at Low) (Low at Low)
        A at B to Mutual

        B at A to None
        B at A to A at B (Low at Low) (High at Low)
        B at A to B at A (Low at Low) (Low at Low)
        B at A to Mutual

        Mutual to None
        Mutual to A at B (High at Low)
        Mutual to B at A (Low at Low)
        Mutual to Mutual
        """

        self.matrix_4x4 = np.zeros((4, 4))
        cm = self.counting_matrix

        self.matrix_4x4 = [
            # None to None
            # None to A at B (High at Low)
            # None to B at A (Low at Low)
            # None to Mutual
            [cm[0, 0, 0, 0], cm[0, 0, 0, 1], cm[0, 1, 0, 0], cm[0, 1, 0, 1]],
            # A at B to None
            # A at B to A at B (High at Low) (High at Low)
            # A at B to B at A (High at Low) (Low at Low)
            # A at B to Mutual
            [cm[0, 0, 1, 0], cm[0, 0, 1, 1], cm[0, 1, 1, 0], cm[0, 1, 1, 1]],
            # B at A to None
            # B at A to A at B (Low at Low) (High at Low)
            # B at A to B at A (Low at Low) (Low at Low)
            # B at A to Mutual
            [cm[1, 0, 0, 0], cm[1, 0, 0, 1], cm[1, 1, 0, 0], cm[1, 1, 0, 1]],
            # Mutual to None
            # Mutual to A at B (High at Low)
            # Mutual to B at A (Low at Low)
            # Mutual to Mutual
            [cm[1, 0, 1, 0], cm[1, 0, 1, 1], cm[1, 1, 1, 0], cm[1, 1, 1, 1]],
        ]
        return np.array(self.matrix_4x4)

    def transform_to_4x4_probabilities_matrix(self):
        """
        translates matrix 2x2x2x2 matrix of [Low_previous_state][Low_current_state][High_previous_state][High_current_state]
        to 4x4 matrix of actions probabilities as stated in .xlsx file

        None to None
        None to A at B (High at Low)
        None to B at A (Low at Low)
        None to Mutual

        A at B to None
        A at B to A at B (High at Low) (High at Low)
        A at B to B at A (High at Low) (Low at Low)
        A at B to Mutual

        B at A to None
        B at A to A at B (Low at Low) (High at Low)
        B at A to B at A (Low at Low) (Low at Low)
        B at A to Mutual

        Mutual to None
        Mutual to A at B (High at Low)
        Mutual to B at A (Low at Low)
        Mutual to Mutual
        """
        cm = self.transform_to_4x4_count_matrix()
        pm = cm / cm.sum(axis=1, keepdims=True)
        return pm
