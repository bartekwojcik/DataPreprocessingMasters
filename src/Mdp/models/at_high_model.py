import numpy as np
from mdp_const import MdpConsts as consts

class AtHighMdpModel:
    """
    mdp model where person at high is an agent and low person is a part of environemt. Action's probabilities are defined from .xlsx file

    States:

    0 - None

    1 - A at B (High at Low)

    2 - B at A (Low at High)

    3 - Mutual


    Per State 4 actions:

    0 - State to None

    1 - State to A at B (State to High at Low)

    2 - State to B at A (State to Low at High)

    3 - State to Mutual

    To maintain flexibility we retain an idea that (s,a) might have stochastic results, therefore each
    Graph[S][A] is a list of tuples (prob_of_going_to_next_state, next_state). (might insert rewards later)

    """

    def __init__(self, probabilities_array:np.ndarray):
        """

        :param probabilities_array: (4x4) array of probabilities
        """

        self.probabilities_array = probabilities_array

        self.states = consts.LIST_OF_STATES
        self.actions = consts.LIST_OF_ACTIONS

        self.graph = {}
        for s in self.states:
            self.graph[s] = {}
            #TODO actions probably should be: Look or not look
            for a in self.actions:
                self.graph[s][a] = []
                for next_s in self.states:
                    prob = self.probabilities_array[s,a]
                    self.graph[s][a].append((prob, next_s))
