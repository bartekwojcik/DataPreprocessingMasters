from typing import List, Tuple

from Mdp.at_high_model_components.at_high_policy_player import HighPolicyPlayer
from settings import Settings
from inverse_reinforcement_learning.conversation_comperar import ConversationComparer
from inverse_reinforcement_learning.irl_processor_result import IrlProcessorResult
import mdp_const


class CompareProcessor:
    """
    Compares real conversation with the reasult created by IRL algorithm
    """
    def compare(
        self,

        irl_result: IrlProcessorResult,
        file_name: str,
        original_conversation: List[dict],
        metadata: dict,
        frame_step: int,
        result_shape: Tuple,
        settings: Settings,
        policy_player: HighPolicyPlayer,
        policy_player_max_step:int,
        show_plot=True,
    ) -> None:
        """
        Compares real conversation with the reasult created by IRL algorithm
        :param irl_result:
        :param file_name:
        :param original_conversation:
        :param metadata:
        :param frame_step:
        :param show_plot:
        :return:
        """

        file_name_to_save_plot = f"{settings.GLOBAL_PREFIX_FOR_FILE_NAMES}" \
                                 f"_{file_name}_frames_{settings.TRANSITION_FRAME_STEP}" \
                                 f"_time_size:{settings.TIME_SIZE}"
        if not irl_result.is_ok:
            file_name_to_save_plot = "FUCKED_UP" + file_name_to_save_plot


        t_values = [t for t, W, Q, policy, reward_matrix in irl_result.list_of_t_W_intercept_policies_rewards]
        policies = [policy for t, W, Q, policy, reward_matrix in irl_result.list_of_t_W_intercept_policies_rewards]

        calculated_conversations = [policy_player.play_policy(policy,policy_player_max_step) for policy in policies]

        comparer = ConversationComparer()
        comparer.compare_and_save_plots(
            file_name_to_save_plot,
            original_conversation=original_conversation,
            calculated_conversations=calculated_conversations,
            t_values = t_values,
            frame_step=frame_step,
            file_metadata=metadata,
            show=show_plot,
            result_shape=result_shape,
            settings=settings
        )
