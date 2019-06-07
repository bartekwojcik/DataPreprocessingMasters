import numpy as np
from settings import Settings
import os
from mdp_const import MdpConsts

import matplotlib.pyplot as plt


def save_plot(file_path, y_label, bins, value_array, state_from, state_to):
    fig = plt.figure()
    fig.suptitle(
        f"from state: {state_from} \n to: {state_to} ", fontsize=14, fontweight="bold"
    )
    plt.bar(bins, value_array, width=1, ec="black")
    plt.ylabel(y_label)
    plt.xlabel("frame step (0.04s)")

    plt.savefig(file_path, dpi=800)
    plt.close(fig)


def plot_histograms(file_name: str, count_array: np.ndarray, settings: Settings):
    n_bins = settings.TIME_SIZE
    bins = np.arange(0, n_bins, 1)
    all_states = MdpConsts.GET_TALK_AND_LOOK_STATES()
    for state_from in all_states:
        for state_to in all_states:
            transition_tuple = state_from + state_to
            times_array = count_array[transition_tuple]
            folder_name = f"frame_{settings.TRANSITION_FRAME_STEP}_time_{settings.TIME_SIZE}"

            folder_path = os.path.join(
                settings.HISTOGRAMS_FOLDER_PATH, folder_name
            )

            if not os.path.exists(folder_path):
                os.mkdir(folder_path)

            file_name_counts = f"{settings.GLOBAL_PREFIX_FOR_FILE_NAMES}_{file_name}_state_" \
                               f"{state_from,state_to}_counts_frame_{settings.TRANSITION_FRAME_STEP}" \
                               f"_time_{settings.TIME_SIZE}.png"
            full_file_name_counts = os.path.join(
                settings.HISTOGRAMS_FOLDER_PATH,folder_name, file_name_counts
            )
            save_plot(
                full_file_name_counts,
                "number of visits",
                bins,
                times_array,
                state_from,
                state_to,
            )

            sum = np.sum(times_array)
            if sum == 0:
                percentage_array = times_array - times_array
            else:
                percentage_array = times_array / sum
            file_name_probas = f"{settings.GLOBAL_PREFIX_FOR_FILE_NAMES}_{file_name}" \
                               f"_state_{state_from,state_to}_probas_frame_" \
                               f"{settings.TRANSITION_FRAME_STEP}_time_{settings.TIME_SIZE}.png"
            full_file_name_probas = os.path.join(
                settings.HISTOGRAMS_FOLDER_PATH, folder_name,file_name_probas
            )

            save_plot(
                full_file_name_probas,
                "chance of transition (%)",
                bins,
                percentage_array,
                state_from,
                state_to,
            )
            debug = 5


if __name__ == "__main__":
    # plot histograms of global data

    settings = Settings()

    file = os.path.join(
        settings.TRANSITION_RESULTS_FOLDER_PATH,
        f"transition_counting_results_with_talk_{settings.TRANSITION_FRAME_STEP}"
        f"_frame_{settings.TIME_SIZE}_time_size.npy",
    )
    count_array = np.load(file)
    plot_histograms("global", count_array,settings)
