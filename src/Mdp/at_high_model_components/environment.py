import random
from typing import Tuple

from Mdp.at_high_model_components.at_high_model import AtHighMdpModel
import numpy as np


class Environment:
    def __init__(self, model: AtHighMdpModel, rewards: np.ndarray):
        """

        :param model:
        :param rewards: np array that is mapping 1-to-1 of model's states
        """
        self.model = model
        self.rewards = rewards
        self.current_state = self.model.states[0]

    @property
    def n_actions(self):
        return len(self.model.actions)

    def reset(self) -> Tuple[int, int, int, int, int]:
        """
        :return: state index
        """
        self.current_state = self.model.states[0]
        return self.current_state

    def step(
        self, action: Tuple[int, int]
    ) -> Tuple[Tuple[int, int, int, int, int], float]:
        """

        :param action: tuple of gaze and talk state, like (1,0) namely: look=1 and not talk=0
        :return: tuple of (new state of the environment, reward for going there)
        """

        possible_environment_responses = self.model.graph[self.current_state][
            action
        ]
        selected_response = self.__select_environment_response(possible_environment_responses)
        response_index = self.model.states.index(selected_response)
        reward = self.rewards[response_index]
        self.current_state = selected_response

        return selected_response, reward


    def __select_environment_response(self, possible_environment_responses):
        """
        THIS FUNCTIONS IS ALMOST THE SAME LIKE POLICY_PLAYER.__random_next_state
        sorry, it is poor coding and lack of time
        :param possible_environment_responses:
        :return:
        """

        rnd = random.random()
        last_proba = 0
        selected_response = -1
        for proba, next_state in possible_environment_responses:
            if rnd < last_proba + proba:
                selected_response = next_state
                return selected_response
            else:
                last_proba += proba

        if last_proba == 0:
            # go to random place
            random_int = random.randint(0, len(possible_environment_responses) - 1)
            return possible_environment_responses[random_int][1]

        assert selected_response != -1, "no state was selected!"

