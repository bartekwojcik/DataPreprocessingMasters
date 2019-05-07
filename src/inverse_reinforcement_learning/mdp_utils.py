from mdp_const import MdpConst


class MdpUtils:
    """
    Bunch of methods for MDP
    """
    states = [MdpConst.AATB, MdpConst.BATA, MdpConst.MUTUAL ,MdpConst.NONE]

    @staticmethod
    def get_state(index):
        return MdpUtils.states[index]
