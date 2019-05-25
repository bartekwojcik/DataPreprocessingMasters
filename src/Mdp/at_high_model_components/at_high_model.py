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
        :param counting_array: (2,2,2,2,2,2,2,2) array of transitions counts
        """

        assert counting_array.shape == (2, 2, 2, 2, 2, 2, 2, 2), "did you update model?"
        self.Ca = counting_array

        self.states = consts.GET_TALK_AND_LOOK_STATES()
        self.actions = consts.GET_TALK_AND_LOOK_ACTIONS()

        self.__init_states()

    def __init_states(self):
        """
        Sets graph where graph[state][action] = List of tuples (probability_of going to this state, state we will end up)

        """
        self.graph = {}
        for s in self.states:
            self.graph[s] = {}
            for a in self.actions:
                self.graph[s][a] = []
                # now collect possible states we might end up given state we are in and action we take
                # get sum of all transition:
                sum = 0
                for p_a in self.actions:
                    # tuple1 + tuple2 = tuple3
                    # and array[0,0] == array[(0,0)]
                    #selc.Ca[s+a+p_a] -> s = state we were in, a = action agent wants to do, p_a = action that the other guy might do
                    sum += self.Ca[s+a + p_a]

                for p_a in self.actions:
                    if sum == 0:
                        proba = 0
                    else:
                        proba = self.Ca[s+ a + p_a] / sum
                    self.graph[s][a].append((proba, a + p_a))
