import numpy as np


class TransitionCountingTranslator:
    def __init__(self, count_matrix: np.ndarray):

        self.counting_matrix = np.array(count_matrix)

        assert self.counting_matrix.shape == (
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
        ), "it is not (2,2,2,2,2,2,2,2) matrix, is it coming from transition_counting_results_with_talk.npy?"

    def transform_to_2D_count_matrix(self)->np.ndarray:
        """
        translates matrix 2x2x2x2 matrix to 16x16
        """

        self.matrix = self.counting_matrix.reshape((16,16))

        return self.matrix

    def transform_to_2D_probabilities_matrix(self):

        with np.errstate(divide='ignore', invalid='ignore'):
            cm = self.transform_to_2D_count_matrix()
            sum = cm.sum(axis=1, keepdims=True)
            # pm = cm / sum

            # https://stackoverflow.com/questions/26248654/how-to-return-0-with-divide-by-zero/32106804#32106804
            c = np.true_divide(cm, sum)
            c[c == np.inf] = 0
            c = np.nan_to_num(c)
            return c
