import numpy as np
from mdp_const import MdpConsts as consts


class AtHighMdpModel:
    """
    mdp model where person at high is an agent and low person is a part of environemt. Action's probabilities are defined from .xlsx file

    To maintain flexibility we retain an idea that (s,a) might have stochastic results, therefore each
    Graph[S][A] is a list of tuples (prob_of_going_to_next_state, next_state).

    """

    def __init__(self, counting_array: np.ndarray):
        """

        :param counting_array: (4x4) array of transitions counts
        """

        self.Ca = counting_array

        self.states = consts.GET_TALK_AND_LOOK_STATES()
        self.actions = consts.GET_TALK_AND_LOOK_ACTIONS()
        
        self.__init_states()
        #TODO HERE IS WORK

        self.graph = {}
        for s in self.states:
            self.graph[s] = {}
            for a in self.actions:
                self.graph[s][a] = []
                # in .xlsx file, part "At high model probas"
                # if a = 0, not look, it always go to "none" =0 or "l at h" =2
                # if a =1, look, it always to go to "h at l" =1 or "mutual" = 3
                first_count = self.Ca[s, a]
                second_count = self.Ca[s, a + 2]
                sum = first_count + second_count
                first_proba = first_count / sum
                second_proba = second_count / sum

                self.graph[s][a].append((first_proba, a))
                self.graph[s][a].append((second_proba, a + 2))

    def __init_states(self):
        self.states = []


