import os

import numpy as np

from Mdp.transition_counting_translator import TransitionCountingTranslator
from mdp_const import MdpConsts as consts
import mdp_const
import settings
from transition_counting.heatmap_plotter import plot_count_heatmap


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

        assert counting_array.shape == (2, 2, 2, 2, 2, 2, 2, 2, mdp_const.TIME_SIZE), "did you update model?"
        self.Ca = counting_array

        self.states = consts.GET_TALK_AND_LOOK_STATES_WITH_TIME()
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
                #it might end at in the same state with time increased by 1
                #OR
                #it might go to another state with time equal to 0

                # get sum of all transition:
                sum = 0
                # because your count_matrix is of shape [high_person_talk, high_person_talk, low_person_gaze, low_person_talk, time]
                # we need to split time from s to append it to action and get the correct format
                current_states_idx = s[:4]
                current_time_idx = (s[4],)

                for p_a in self.actions:
                    # tuple1 + tuple2 = tuple3
                    # and array[0,0] == array[(0,0)]
                    # selc.Ca[s+a+p_a] -> s = state we were in, a = action agent wants to do, p_a = action that the other guy might do

                    sum += self.Ca[current_states_idx + a + p_a + current_time_idx]

                for p_a in self.actions:
                    if sum == 0:
                        proba = 0
                    else:
                        proba = self.Ca[current_states_idx + a + p_a + current_time_idx] / sum

                    new_state_idx = (a+p_a)
                    #if old state is the same as new one:
                    if current_states_idx == new_state_idx:
                        #update time by one

                        if s[4]+1 < mdp_const.TIME_SIZE:
                            next_time = s[4]+1
                        else:
                            next_time = 0
                        self.graph[s][a].append((proba, a + p_a+(next_time,)))
                    else:
                        #we are going to new state and we are going to time step 0
                        self.graph[s][a].append((proba, a + p_a + (0,)))
        debug = 5


    def plot_probabilities_per_state(self,verbose:bool,file_name:str):

        results = np.zeros_like(self.Ca)

        for s in self.states:
            for a in self.actions:
                debug = 5
                s_wo_time = s[:4]
                time = (s[4],)
                for proba, next_state in self.graph[s][a]:
                    next_s_wo_time = next_state[:4]

                    results[s_wo_time + next_s_wo_time + time] += proba

        translator = TransitionCountingTranslator(results)
        #probabilities_matrix = translator.transform_to_2D_probabilities_matrix()
        count_matrix = translator.transform_to_2D_count_matrix()

        file_name_count = os.path.join(settings.COMPARISON_PLOTS_FOLDER_PATH,f"{file_name}_MDP_MODEL_GRAPTH_counts.jpg")
        #file_name_proba = os.path.join(settings.COMPARISON_PLOTS_FOLDER_PATH,f"{file_name}_MDP_MODEL_GRAPTH_probas.jpg")
        plot_count_heatmap(np.round(count_matrix, decimals=2),file_name_count,show=verbose)
        #plot_count_heatmap(probabilities_matrix,file_name_proba,show=verbose)