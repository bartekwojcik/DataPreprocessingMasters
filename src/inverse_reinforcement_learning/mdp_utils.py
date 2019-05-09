import os
from typing import Tuple

import numpy as np
from inverse_reinforcement_learning.mdp_components import MdpAction, MdpState, MdpGraph
from mdp_const import MdpConsts as consts
import settings
import itertools


class Simple16ActionMdp:
    """
    Sets graph of MDP with 16 actions with their probabilities in variable graph
    """

    def __init__(self, counts: np.ndarray):
        """

        :param counts: array of transition counts (probably from file "transition_counting_results")
        """
        atob_sum = (
            counts[0, 0, 1, 1]
            + counts[0, 1, 1, 0]
            + counts[0, 0, 1, 0]
            + counts[0, 1, 1, 1]
        )
        AToBState = MdpState(
            consts.AATB,
            [
                MdpAction(
                    consts.AATB_TO_AATB, {consts.AATB: counts[0, 0, 1, 1] / atob_sum}
                ),
                MdpAction(
                    consts.AATB_TO_BATA, {consts.BATA: counts[0, 1, 1, 0] / atob_sum}
                ),
                MdpAction(
                    consts.AATB_TO_NONE, {consts.NONE: counts[0, 0, 1, 0] / atob_sum}
                ),
                MdpAction(
                    consts.AATB_TO_MUTUAL,
                    {consts.MUTUAL: counts[0, 1, 1, 1] / atob_sum},
                ),
            ],
        )

        b_to_a_sum = (
            counts[1, 0, 0, 1]
            + counts[1, 1, 0, 0]
            + counts[1, 0, 0, 0]
            + counts[1, 1, 0, 1]
        )
        BToAState = MdpState(
            consts.BATA,
            [
                MdpAction(
                    consts.BATA_TO_AATB, {consts.AATB: counts[1, 0, 0, 1] / b_to_a_sum}
                ),
                MdpAction(
                    consts.BATA_TO_BATA, {consts.BATA: counts[1, 1, 0, 0] / b_to_a_sum}
                ),
                MdpAction(
                    consts.BATA_TO_NONE, {consts.NONE: counts[1, 0, 0, 0] / b_to_a_sum}
                ),
                MdpAction(
                    consts.BATA_TO_MUTUAL,
                    {consts.MUTUAL: counts[1, 1, 0, 1] / b_to_a_sum},
                ),
            ],
        )

        none_sum = (
            counts[0, 0, 0, 1]
            + counts[0, 1, 0, 0]
            + counts[0, 0, 0, 0]
            + counts[0, 1, 0, 1]
        )
        NoneState = MdpState(
            consts.NONE,
            [
                MdpAction(
                    consts.NONE_TO_AATB, {consts.AATB: counts[0, 0, 0, 1] / none_sum}
                ),
                MdpAction(
                    consts.NONE_TO_BATA, {consts.BATA: counts[0, 1, 0, 0] / none_sum}
                ),
                MdpAction(
                    consts.NONE_TO_NONE, {consts.NONE: counts[0, 0, 0, 0] / none_sum}
                ),
                MdpAction(
                    consts.NONE_TO_MUTUAL,
                    {consts.MUTUAL: counts[0, 1, 0, 1] / none_sum},
                ),
            ],
        )

        mutual_sum = (
            counts[1, 0, 1, 1]
            + counts[1, 1, 1, 0]
            + counts[1, 0, 1, 0]
            + counts[1, 1, 1, 1]
        )
        MutualState = MdpState(
            consts.MUTUAL,
            [
                MdpAction(
                    consts.MUTUAL_TO_AATB,
                    {consts.AATB: counts[1, 0, 1, 1] / mutual_sum},
                ),
                MdpAction(
                    consts.MUTUAL_TO_BATA,
                    {consts.BATA: counts[1, 1, 1, 0] / mutual_sum},
                ),
                MdpAction(
                    consts.MUTUAL_TO_NONE,
                    {consts.NONE: counts[1, 0, 1, 0] / mutual_sum},
                ),
                MdpAction(
                    consts.MUTUAL_TO_MUTUAL,
                    {consts.MUTUAL: counts[1, 1, 1, 1] / mutual_sum},
                ),
            ],
        )

        graph = MdpGraph([MutualState, NoneState, AToBState, BToAState])
        self.graph = graph

    def get_states_names(self):
        return list(map(lambda state: state.name, list(self.get_states())))

    def get_states(self):
        return self.graph.states

    def get_actions_names(self):
        actions_names = [action.name for action in itertools.chain(*self.get_actions())]
        return actions_names #list(map(lambda action: action.name, actions))

    def get_actions(self):
        for state in self.graph.states:
            yield state.actions


class MdpUtils:
    __Simple16ActionMdp = None  # type: Simple16ActionMdp

    @staticmethod
    def simple_16_action_graph() -> Simple16ActionMdp:
        """
        Gets MDP model from "transition_counting_results.npy" file
        :return: MDP model as graph.
        """
        if MdpUtils.__Simple16ActionMdp:
            return MdpUtils.__Simple16ActionMdp

        else:
            file = os.path.join(
                settings.MY_DATA_FOLDER_PATH, "transition_counting_results.npy"
            )
            array = np.load(file)
            MdpUtils.__Simple16ActionMdp = Simple16ActionMdp(array)
            return MdpUtils.__Simple16ActionMdp

    @staticmethod
    def get_state(main_state: int, other_state: int) -> str:
        """

        :param main_state: 0 if not looking at person2, 1 if looking at person 2
        :param other_state: 0 if not looking at person1, 1 if looking at person 1
        :return: name of state
        """

        model = MdpUtils.simple_16_action_graph()

        if main_state == 0 and other_state == 0:
            state = next(
                state.name for state in model.graph.states if state.name == consts.NONE
            )
            return state

        elif main_state == 1 and other_state == 1:
            state = next(
                state.name
                for state in model.graph.states
                if state.name == consts.MUTUAL
            )
            return state

        elif main_state == 1 and other_state == 0:
            state = next(
                state.name for state in model.graph.states if state.name == consts.AATB
            )
            return state

        elif main_state == 0 and other_state == 1:
            state = next(
                state.name for state in model.graph.states if state.name == consts.BATA
            )
            return state

        else:
            raise ValueError("no combination matches")

    @staticmethod
    def get_states_from_action_name(action_name:str, list_of_states = consts.LIST_OF_STATES)->Tuple[str,str]:
        """
        Assumes " to " string does not exist in States names
        :param action_name: string for instance "A to B to Mutual"
        :return: tuple for instance ("A to B", "Mutual")
        """

        original_action_name = action_name

        splitted_action = action_name.split(" to ")
        start_state, end_state = splitted_action[0], splitted_action[1]

        assert start_state
        assert end_state
        return start_state,end_state


