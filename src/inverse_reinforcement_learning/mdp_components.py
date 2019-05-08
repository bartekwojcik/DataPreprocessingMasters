from typing import List


class MdpAction:
    def __init__(self, name, state_it_leads_to: dict):
        """
        :param name: name
        :param state_it_leads_to: dictionary [name_of_state: probability_of_choosing]
        """
        self.state_it_leads_to = state_it_leads_to
        self.name = name


class MdpState:
    def __init__(self, name, actions: List[MdpAction]):
        self.name = name
        self.actions = actions


class MdpGraph:
    def __init__(self, states: List[MdpState]):
        self.states = states
