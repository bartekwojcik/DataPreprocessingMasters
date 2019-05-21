from typing import List

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
        show_plot=True,
    ) -> None:
        file_name_to_save_plot = file_name
        if not irl_result.is_ok:
            file_name_to_save_plot = "FUCKED_UP" + file_name_to_save_plot

        comparer = ConversationComparer()
        comparer.compare_and_save_plots(
            file_name_to_save_plot,
            original_conversation=original_conversation,
            calculated_conversation=irl_result.new_conversation,
            frame_step=frame_step,
            file_metadata=metadata,
            show=show_plot,
        )
