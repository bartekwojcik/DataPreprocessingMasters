
class IrlProcessorResult:
    def __init__(self,weights, reward_matrix, policy, V, new_conversation, is_ok):
        self.V = V
        self.weights = weights
        self.reward_matrix = reward_matrix
        self.policy = policy
        self.new_conversation = new_conversation
        self.is_ok = is_ok