from inverse_reinforcement_learning.main import main_synchronous as irl
from kl_divergence.main import plot_kullback_leibler
from settings import Settings
import os


"""
THIS IS AN EXAMPLE OF HOW TO USE ALL THE CODE IN THIS PROJECT.
TO USE IT CORRECTLY YOU NEED TO HAVE THE SAME FOLDER STRUCTURE OF "original-data" AND PART OF "my-data" AS IT IS HERE.
YOU CAN MANIPULATE THIS BY SETTING PATHS IN Settings() OBJECT TO FOLDERS OF YOUR CHOOSING. 
inverse_reinforcement_learning.main.main_synchronous PRODUCES OUTPUT IN DESIRED FOLDER (root_folder_for_plots) AND 
plot_kullback_leibler USES THIS OUTPUT (WORK_FOLDER_PATH_WITH_POLICIES), NAMEPY IT EXPECTS ALL THESE .NPY FILES TO BE THERE.
"""

VERBOSE = False  # if true shows all intermediate plots
global_settings = Settings(MAX_CONTINUOUS_TIME_SEC=10,CONTINUOUS_TIME_STEP_SEC=0.04)  # stores paths to folders and settings

root_folder_for_plots = (
    global_settings.COMPARISON_PLOTS_FOLDER_PATH
)  # defines folder where outputs per conversation will be stored
HUMAN_READABLE_FOLDER_PATH = global_settings.HUMAN_READABLE_FOLDER_PATH #this is default path
METADATA_PATH = global_settings.READABLE_METADATA_FILE_PATH #this is default path
TRANSITION_FRAME_STEP = global_settings.TRANSITION_FRAME_STEP #this is default path
TIME_SIZE = global_settings.TIME_SIZE #this paramter is calculated with Settings(MAX_CONTINUOUS_TIME_SEC=10,CONTINUOUS_TIME_STEP_SEC=0.04)

irl(
    VERBOSE,
    root_folder_for_plots,
    HUMAN_READABLE_FOLDER_PATH,
    METADATA_PATH,
    TRANSITION_FRAME_STEP,
    TIME_SIZE,
)  # performs inverse reinforcement learning algorithm on all files in settings.HUMAN_READABLE_FOLDER_PATH supported with settings.READABLE_METADATA_FILE_PATH

#WORK_FOLDER_PATH_WITH_POLICIES point it to the folder where "human_readable_conversation_XX.json_policies.npy" results are. It needs other .npy files that the results of inverse_reinforcement_learning.main
WORK_FOLDER_PATH_WITH_POLICIES = "C:\\Users\\kicjo\\Documents\\PythonProjects\\DataPreprocessing-Masters\\my-data\\human_readable_conversation_11.json\\policies"
#folder where histograms will be saved
HISTOGRAMS_FOLDER_PATH = os.path.join(WORK_FOLDER_PATH_WITH_POLICIES, "histograms")

plot_kullback_leibler(
    WORK_FOLDER_PATH_WITH_POLICIES,
    HISTOGRAMS_FOLDER_PATH,
    TIME_SIZE,
    TRANSITION_FRAME_STEP,
    global_settings.READABLE_METADATA_FILE_PATH,
    global_settings.HUMAN_READABLE_FOLDER_PATH,
)
