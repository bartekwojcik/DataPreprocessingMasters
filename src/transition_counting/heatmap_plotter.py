import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os


def plot_count_heatmap(array: np.ndarray, file_path_to_save: str, show: bool = True) -> None:
    """
    Taken from matplotlib website
    :param array:
    :param file_path_to_save:
    :return:
    """
    states = ["None", "High at Low", "Low at High", "Mutual"]

    fig, ax = plt.subplots()
    im = ax.imshow(array)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(states)))
    ax.set_yticks(np.arange(len(states)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(states)
    ax.set_yticklabels(states)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(states)):
        for j in range(len(states)):
            text = ax.text(j, i, array[i, j],
                           ha="center", va="center", color="w")

    base = os.path.basename(file_path_to_save)
    ax.set_title(base)
    fig.tight_layout()
    fig.set_size_inches((20, 10))
    plt.savefig(file_path_to_save, quality=70, dpi=400)
    if show:
        plt.show()
    else:
        plt.close(fig)