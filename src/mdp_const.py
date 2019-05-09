def create_action_name(state_from, state_to):
    return f"{state_from} to {state_to}"


class MdpConsts:

    NONE = "None"
    AATB = "A at B"
    BATA = "B at A"
    MUTUAL = "Mutual"

    LIST_OF_STATES = [NONE, AATB, BATA, MUTUAL]

    AATB_TO_NONE = create_action_name(AATB, NONE)
    AATB_TO_AATB = create_action_name(AATB, AATB)
    AATB_TO_BATA = create_action_name(AATB, BATA)
    AATB_TO_MUTUAL = create_action_name(AATB, MUTUAL)

    BATA_TO_NONE = create_action_name(BATA, NONE)
    BATA_TO_AATB = create_action_name(BATA, AATB)
    BATA_TO_BATA = create_action_name(BATA, BATA)
    BATA_TO_MUTUAL = create_action_name(BATA, MUTUAL)

    MUTUAL_TO_NONE = create_action_name(MUTUAL, NONE)
    MUTUAL_TO_AATB = create_action_name(MUTUAL, AATB)
    MUTUAL_TO_BATA = create_action_name(MUTUAL, BATA)
    MUTUAL_TO_MUTUAL = create_action_name(MUTUAL, MUTUAL)

    NONE_TO_NONE = create_action_name(NONE, NONE)
    NONE_TO_AATB = create_action_name(NONE, AATB)
    NONE_TO_BATA = create_action_name(NONE, BATA)
    NONE_TO_MUTUAL = create_action_name(NONE, MUTUAL)

    LIST_OF_ACTIONS = [
        AATB_TO_NONE,
        AATB_TO_MUTUAL,
        AATB_TO_BATA,
        AATB_TO_AATB,
        NONE_TO_AATB,
        NONE_TO_BATA,
        NONE_TO_MUTUAL,
        NONE_TO_NONE,
        BATA_TO_AATB,
        BATA_TO_BATA,
        BATA_TO_MUTUAL,
        BATA_TO_NONE,
        MUTUAL_TO_AATB,
        MUTUAL_TO_BATA,
        MUTUAL_TO_MUTUAL,
        MUTUAL_TO_NONE,
    ]
