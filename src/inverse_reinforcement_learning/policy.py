import json


class Policy:
    """
    Describes policy
    """

    def __init__(self, data, n_states, n_actions) -> None:
        self.n_actions = n_actions
        self.n_states = n_states
        self.data = data

