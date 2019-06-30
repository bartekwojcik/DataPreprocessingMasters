from typing import List, Tuple

from Mdp.at_high_model_components.at_high_policy_player import HighPolicyPlayer
from inverse_reinforcement_learning.conversation_comperar import ConversationComparer
from inverse_reinforcement_learning.irl_processor_result import IrlProcessorResult


class CompareProcessor:
    def compare(
        self,
        irl_result: IrlProcessorResult,
        file_name: str,
        original_conversation: List[dict],
        metadata: dict,
        frame_step: int,
            result_shape: Tuple,
            policy_player: HighPolicyPlayer,
        show_plot=True
    ) -> None:

        file_name_to_save_plot = f"co_{file_name}"


        if not irl_result.is_ok:
            file_name_to_save_plot = "FUCKED_UP" + file_name_to_save_plot

        t_values = [t for t, W, Q, policy, reward_matrix in irl_result.list_of_ts]
        policies = [policy for t, W, Q, policy, reward_matrix in irl_result.list_of_ts]

        calculated_conversations = [policy_player.play_policy(policy, len(original_conversation)) for policy in policies]

        comparer = ConversationComparer()
        comparer.compare_and_save_plots(
            file_name_to_save_plot,
            original_conversation=original_conversation,
            calculated_conversations=calculated_conversations,
            t_values=t_values,
            frame_step=frame_step,
            file_metadata=metadata,
            show=show_plot,
            result_shape=result_shape

        )
