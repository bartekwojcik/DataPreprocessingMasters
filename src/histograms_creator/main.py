import numpy as np
import settings
import os
from mdp_const import MdpConsts
from Mdp.mdp_utils import MdpUtils
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


def plot_histograms(file_name:str, count_array:np.ndarray):
    n_bins = MdpConsts.TIME_SIZE
    bins = np.arange(0, n_bins, 1)
    all_states = MdpConsts.GET_TALK_AND_LOOK_STATES()
    for state_from in all_states:
        for state_to in all_states:
            transition_tuple = state_from + state_to
            times_array = count_array[transition_tuple]

            fig = plt.figure()
            fig.suptitle(f'from state: {state_from} \n to: {state_to} ', fontsize=14, fontweight='bold')
            plt.bar(bins, times_array, width=1, ec='black')
            plt.ylabel('number of visits')
            plt.xlabel('frame step (0.04s)')
            file_name = f"{settings.GLOBAL_PREFIX_FOR_FILE_NAMES}_{file_name}_frame_{settings.TRANSITION_FRAME_STEP}_time_{MdpConsts.TIME_SIZE}.png"
            full_file_name = os.path.join(settings.HISTOGRAMS_FOLDER_PATH, file_name)
            plt.savefig(full_file_name, dpi=800)
            plt.close(fig)
            debu = 5


if __name__ == '__main__':
    #plot histograms of global data

    file = os.path.join(
        settings.TRANSITION_RESULTS_FOLDER_PATH,
        f"transition_counting_results_with_talk_{settings.TRANSITION_FRAME_STEP}_frame_{MdpConsts.TIME_SIZE}_time_size.npy"
    )
    count_array = np.load(file)
    plot_histograms("global",count_array)







