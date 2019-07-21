import numpy as np
import mdp_const
from settings import Settings


class TransitionCountingTranslator:
    def __init__(self, count_matrix: np.ndarray, maximum_time_size:int):

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
            maximum_time_size
        ), "counting matrix shape is nor correct (check if the time, last index, is correct)"

    def transform_to_2D_count_matrix(self)->np.ndarray:
        """
        translates matrix 2x2x2x2 matrix to 16x16
        """
        #sum the time dimentions, so it looks like there was no time dimentions and have just sums per state transitions

        sums_per_time = self.counting_matrix.sum(axis=-1)
        matrix = sums_per_time.reshape((16,16))

        return matrix

    def transform_to_2D_probabilities_matrix(self):

        with np.errstate(divide='ignore', invalid='ignore'):
            cm = self.transform_to_2D_count_matrix()
            sumation = cm.sum(axis=1, keepdims=True)
            # pm = cm / sum

            # https://stackoverflow.com/questions/26248654/how-to-return-0-with-divide-by-zero/32106804#32106804
            c = np.true_divide(cm, sumation)
            c[c == np.inf] = 0
            c = np.nan_to_num(c)
            return c
