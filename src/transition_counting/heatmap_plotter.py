import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def plot_count_heatmap(array:np.ndarray, file_path_to_save:str)->None:
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

    ax.set_title("State to state transitions (number)")
    fig.tight_layout()
    plt.savefig(file_path_to_save)
    plt.show()
