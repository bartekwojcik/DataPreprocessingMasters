import itertools

class MdpConsts:


    NOT_LOOK = 0
    LOOK = 1

    QUIET = 0
    TALK = 1
    __LIST_OF_LOOKING_STATES = [NOT_LOOK, LOOK]
    __LIST_OF_TALKING_STATES = [QUIET, TALK]

    @classmethod
    def GET_TALK_AND_LOOK_STATES_WITH_TIME(cls, time_size:int):
        """
        :return: List of tuples - possible states [(High_gaze,High_talk,Low_gaze,Low_talk,time0)...]
        """

        result = []
        for hg in cls.__LIST_OF_LOOKING_STATES:
            for ht in cls.__LIST_OF_TALKING_STATES:
                for lg in cls.__LIST_OF_LOOKING_STATES:
                    for lt in cls.__LIST_OF_TALKING_STATES:
                        # so i am not interested in real time (0.04 sec etc but in RELATIVE TIME STEP IN CONVERSATION
                        for time in range(time_size):
                            result.append((hg, ht, lg, lt, time))

        return result

    @classmethod
    def GET_TALK_AND_LOOK_STATES(cls):
        """
        :return: List of tuples - possible states [(High_gaze,High_talk,Low_gaze,Low_talk,time0)...]
        """

        result = []
        for hg in cls.__LIST_OF_LOOKING_STATES:
            for ht in cls.__LIST_OF_TALKING_STATES:
                for lg in cls.__LIST_OF_LOOKING_STATES:
                    for lt in cls.__LIST_OF_TALKING_STATES:
                        result.append((hg, ht, lg, lt))

        return result

    @classmethod
    def GET_TALK_AND_LOOK_ACTIONS(cls):
        # maybe instead of tuple i should use list?
        result = list(tuple for tuple in itertools.product(cls.__LIST_OF_LOOKING_STATES, cls.__LIST_OF_TALKING_STATES))
        return result
