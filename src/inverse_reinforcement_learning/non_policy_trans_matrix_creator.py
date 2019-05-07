from inverse_reinforcement_learning.policy import Policy
from typing import List, Tuple


class NonPolicyMatrixCreator:

    def __init__(self, optimal_policies: List[Policy])-> None:
        self._optimal_policies = optimal_policies
