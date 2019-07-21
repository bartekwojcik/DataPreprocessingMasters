import numpy as np
import os
from mdp_const import MdpConsts
from transition_counting.state_utils import StateUtils

import matplotlib.pyplot as plt


def save_plot(file_path, y_label, bins, value_array, state_from, state_to):
    fig = plt.figure()

    str_state_from = StateUtils.state_vector_to_human_string(state_from)
    str_state_to = StateUtils.state_vector_to_human_string(state_to)

    fig.suptitle(
        f"from {str_state_from} \n to {str_state_to}", fontsize=14, fontweight="bold"
    )
    plt.bar(bins, value_array, width=1, ec="black")
    plt.ylabel(y_label)
    plt.xlabel("frame step (0.04s)")
    fig.set_size_inches((14, 7), forward=False)
    plt.savefig(file_path, quality=10, dpi=200)
    plt.close(fig)


def plot_histograms(conversation_file_name: str, count_array: np.ndarray, n_bins:int, folder_to_save_path:str, histogram_file_name):
    n_bins = n_bins#settings.TIME_SIZE
    bins = np.arange(0, n_bins, 1)
    all_states = MdpConsts.GET_TALK_AND_LOOK_STATES()
    for state_from in all_states:
        for state_to in all_states:
            transition_tuple = state_from + state_to
            times_array = count_array[transition_tuple]
            summation = np.sum(times_array)

            if np.all(times_array == 0) or summation < 5:
                continue

            #folder_name = f"frame_{settings.TRANSITION_FRAME_STEP}_time_{settings.TIME_SIZE}"

            if not os.path.exists(folder_to_save_path):
                os.makedirs(folder_to_save_path)

            frequency = int(np.sum(times_array))

            file_name_counts = f"{frequency}_conv_{conversation_file_name}_{histogram_file_name}_{state_from}_{state_to}_counts.png"
            full_file_name_counts = os.path.join(
                folder_to_save_path, file_name_counts
            )
            save_plot(
                full_file_name_counts,
                "number of visits",
                bins,
                times_array,
                state_from,
                state_to,
            )

            sumation = np.sum(times_array)
            if sumation == 0:
                percentage_array = times_array - times_array
            else:
                percentage_array = times_array / sumation

            file_name_probas = f"{frequency}_conv_{conversation_file_name}_{histogram_file_name}_{state_from}_{state_to}_probas.png"
            full_file_name_probas = os.path.join(
                folder_to_save_path,file_name_probas
            )

            save_plot(
                full_file_name_probas,
                "chance of transition (%)",
                bins,
                percentage_array,
                state_from,
                state_to,
            )

