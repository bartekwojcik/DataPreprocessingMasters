import os

import numpy as np

from Mdp.models.simple_16_deterministic_model import Simple16ActionMdpModel
from mdp_const import MdpConsts as consts
import settings


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
