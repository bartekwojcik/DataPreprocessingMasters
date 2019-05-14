from typing import Tuple


class AtHightAtLowCalculator:
    """
    Saves information about given file with regard to whom looked at whom for longer (stated in percentages)
    """
    def __init__(self, file_name:str):
        self.file_name = file_name
        self.person1_looking_states_count = 0
        self.person2_looking_states_count = 0
        self.all_states_count = 0

    def update(self, person1_state: int, person2_state: int) -> None:
        """
        :param person1_state: 0 - not looking at person 2, 1 - looking
        :param person2_state: 0 - not looking at person 1, 1 - looking
        """
        self.person1_looking_states_count += person1_state
        self.person2_looking_states_count += person2_state
        self.all_states_count += 1

    def get_results(self) -> Tuple[str, float, float]:
        """
        :return: Tuple of percentages of (file_name, person1 looked at person2: float, person2 looked at person1: float)
        """
        # person1 looked at person2 X% of the time
        p1_at_p2 = self.person1_looking_states_count / self.all_states_count

        # person2 looked at person1 X% of the time
        p2_at_p1 = self.person2_looking_states_count / self.all_states_count
        return self.file_name, p1_at_p2, p2_at_p1
