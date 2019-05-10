import numpy as np
from typing import List

from inverse_reinforcement_learning.mdp_utils import MdpUtils


class PolicyMatrixCreator:

    def __init__(self, optimal_policy: List[dict],all_actions:List[str], all_states:List[str])-> None:
        """

        :param optimal_policy: expert's policy in form of list[{state:action}]
        :param all_actions: all actions available for agent
        :param all_states: all actions that agent might be in
        """
        self.all_states = all_states
        self.all_actions = all_actions
        self._optimal_policy = optimal_policy
        x = self.__get_naive_optimal_policy_transition_probas()
        debug = 5

    def __get_naive_policy_transition_probas(self):
        """

        produces optimal's policy transition matrix
        Warning: It assumes each action go to only one state and has probability of one
        :return:
        """
        n_actions = len(self.all_actions)
        n_states = len(self.all_states)

        transition_matrix = np.zeros((n_states,n_states))
        counter_matrix = np.zeros_like(transition_matrix)

        non_policy_matrixes = self.__initialize_non_policy_matrices()

        for state_action in self._optimal_policy:
            initial_state = list(state_action.keys())[0]
            current_action = state_action[initial_state]
            next_state = MdpUtils.get_states_from_action_name(current_action)[1]

            init_state_index = self.all_states.index(initial_state)
            end_state_index = self.all_states.index(next_state)

            counter_matrix[init_state_index,end_state_index]+= 1

            for action_index, non_policy_action in enumerate(self.all_actions):
                if non_policy_action == current_action:
                    continue
                # TODO actually, which matrix is updated when in the same state (but later in time) different action is taken then it was at the first time we were in the same state?




        counter_matrix = np.divide(counter_matrix,counter_matrix,out= np.zeros_like(counter_matrix),where=counter_matrix!=0)
        debug = 5

    def __initialize_non_policy_matrices(self)-> np.ndarray:
        """

        :return: (N-1,M,M)dim matrix where N is number of available actions and M is number of states
        """
        n_actions = len(self.all_actions)
        n_states = len(self.all_states)
        matrices = np.zeros((n_actions-1,n_states,n_states))
        return  matrices





