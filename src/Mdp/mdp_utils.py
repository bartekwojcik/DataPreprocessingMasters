import os

import numpy as np
from mdp_const import MdpConsts as consts
import settings


class Simple16ActionMdpModel:
    """
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
    Graph[S][A] is a list of tuples (prob_of_going_to_next_state, next_state). Insert reward here when known

    """

    def __init__(self):
        self.states = consts.LIST_OF_STATES
        self.actions = consts.LIST_OF_ACTIONS

        # self.graph = {
        #     consts.NONE : {
        #         consts.STATE_TO_NONE: [{}],
        #         consts.STATE_TO_H_AT_L: [{}],
        #         consts.STATE_TO_L_AT_H: [{}],
        #         consts.STATE_TO_MUTUAL: [{}]
        #                    },
        #     consts.H_AT_L:{
        #         consts.STATE_TO_NONE: [{}],
        #         consts.STATE_TO_H_AT_L: [{}],
        #         consts.STATE_TO_L_AT_H: [{}],
        #         consts.STATE_TO_MUTUAL: [{}]
        #     },
        #     consts.L_AT_H:{
        #         consts.STATE_TO_NONE: [{}],
        #         consts.STATE_TO_H_AT_L: [{}],
        #         consts.STATE_TO_L_AT_H: [{}],
        #         consts.STATE_TO_MUTUAL: [{}]
        #     },
        #     consts.MUTUAL:{
        #         consts.STATE_TO_NONE: [{}],
        #         consts.STATE_TO_H_AT_L: [{}],
        #         consts.STATE_TO_L_AT_H: [{}],
        #         consts.STATE_TO_MUTUAL: [{}]
        #     }
        # }

        self.graph = {}
        for s in self.states:
            self.graph[s] = {}
            for a in self.actions:
                self.graph[s][a] = [(1, a)]


class MdpUtils:
    __Simple16ActionMdp = None  # type: Simple16ActionMdpModel

    @staticmethod
    def simple_16_action_graph() -> Simple16ActionMdpModel:
        """
        Gets MDP model from "transition_counting_results.npy" file
        :return: MDP model as graph.
        """
        if MdpUtils.__Simple16ActionMdp:
            return MdpUtils.__Simple16ActionMdp

        else:
            # TODO NOT USED NOW, left here on purpose
            if False:
                file = os.path.join(
                    settings.MY_DATA_FOLDER_PATH, "transition_counting_results.npy"
                )
                array = np.load(file)

            MdpUtils.__Simple16ActionMdp = (
                Simple16ActionMdpModel()
            )  # Simple16ActionMdpModel(array)
            return MdpUtils.__Simple16ActionMdp

    @staticmethod
    def get_state(high_state: int, low_state: int):
        """
        Returns state of given configuration
        :param high_state: gaze state of person at high, 0 - not looking, 1 - looking
        :param low_state: gaze state of person at low, 0 - not looking, 1 - looking
        :return: integer symbolising state ( 0 - None
                1 - A at B (High at Low)
                2 - B at A (Low at High)
                3 - Mutual)
        """

        if high_state == 0 and low_state == 0:
            return consts.NONE
        elif high_state == 1 and low_state == 0:
            return consts.H_AT_L
        elif high_state == 0 and low_state == 1:
            return consts.L_AT_H
        elif high_state == 1 and low_state == 1:
            return consts.MUTUAL
        else:
            raise ValueError(
                f"No combination of gaze states matches: H:{high_state}, L:{low_state}"
            )

    @staticmethod
    def get_action(first_state: int, end_state: int):
        """
        Returns state of given configuration. To be honest, given my current code configuration (15.05.2019), number of action is corresponding to end_state
        :param first_state: previous state of the model, NONE, MUTUAl etc
        :param end_state: end state of the model, NONE, MUTUAl etc
        :return: integer symbolising action ( 0 - State to None
              1 - State to A at B (High at Low)
              2 - State to B at A (Low at High)
              3 - State to Mutual)
        """

        graph = MdpUtils.simple_16_action_graph().graph

        s1 = graph[first_state]
        action = next(action for action in s1.values() if action[0][1] == end_state)

        return action[1]
