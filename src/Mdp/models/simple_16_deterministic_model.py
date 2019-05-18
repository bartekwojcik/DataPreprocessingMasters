from mdp_const import MdpConsts as consts


class Simple16ActionMdpModel:
    """
    States:

    0 - None

    1 - A at B (High at Low)

    2 - B at A (Low at High)

    3 - Mutual


    Per State 4 actions:

    0 - State to None

    1 - State to A at B (State to High at Low)

    2 - State to B at A (State to Low at High)

    3 - State to Mutual

    To maintain flexibility we retain an idea that (s,a) might have stochastic results, therefore each
    Graph[S][A] is a list of tuples (prob_of_going_to_next_state, next_state). Insert reward here when known

    """

    def __init__(self):
        self.states = consts.LIST_OF_STATES
        self.actions = consts.LIST_OF_ACTIONS

        # self.graph = {
        #     consts.NONE : {
        #         consts.STATE_TO_NONE: [{}],
        #         consts.STATE_TO_H_AT_L: [{}],
        #         consts.STATE_TO_L_AT_H: [{}],
        #         consts.STATE_TO_MUTUAL: [{}]
        #                    },
        #     consts.H_AT_L:{
        #         consts.STATE_TO_NONE: [{}],
        #         consts.STATE_TO_H_AT_L: [{}],
        #         consts.STATE_TO_L_AT_H: [{}],
        #         consts.STATE_TO_MUTUAL: [{}]
        #     },
        #     consts.L_AT_H:{
        #         consts.STATE_TO_NONE: [{}],
        #         consts.STATE_TO_H_AT_L: [{}],
        #         consts.STATE_TO_L_AT_H: [{}],
        #         consts.STATE_TO_MUTUAL: [{}]
        #     },
        #     consts.MUTUAL:{
        #         consts.STATE_TO_NONE: [{}],
        #         consts.STATE_TO_H_AT_L: [{}],
        #         consts.STATE_TO_L_AT_H: [{}],
        #         consts.STATE_TO_MUTUAL: [{}]
        #     }
        # }

        self.graph = {}
        for s in self.states:
            self.graph[s] = {}
            for a in self.actions:
                self.graph[s][a] = [(1, a)]