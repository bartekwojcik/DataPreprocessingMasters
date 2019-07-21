from concurrent.futures import ProcessPoolExecutor

from Mdp.at_high_model_components.at_high_policy_player import HighPolicyPlayer
from inverse_reinforcement_learning.compare_processor import CompareProcessor
from inverse_reinforcement_learning.get_model_probas import ModelProbasGetter
from inverse_reinforcement_learning.irl_processor import IrlProcessor
from inverse_reinforcement_learning.irl_results_plotter_saver import (
    IrlResultsPlotterSaver,
)


def process_file(
    metadata_json, filename,
        conv_json,
        full_file_name,
        VERBOSE,
        q_learning_episode_length:int,
        transition_frame_step:int,
        maximum_time_size_frames:int,
        Q_ITERATIONS,
        DISCOUNT_FACTOR,
        Q_ALPHA,
        Q_EPSILON,
        IRL_SOLVER_EPSILON,
        heatmap_folder_path,
        policies_save_folder_path
):
    print(f"file started: {full_file_name} ")

    this_file_metadata = metadata_json[filename]

    probas_getter = ModelProbasGetter()
    mdp_graph = probas_getter.get_model_probas(
        conv_json,
        this_file_metadata,
        transition_frame_step,
        maximum_time_size_frames,
    )

    # plots probabilities of model's actions in states
    #mdp_graph.plot_probabilities_per_state(VERBOSE,filename, settings)

    data_length = len(conv_json)
    IRL_SOLVER_ITERATIONS = 3

    policy_player = HighPolicyPlayer(this_file_metadata, mdp_graph, 0.05)

    processor = IrlProcessor()
    irl_result = processor.process(
        conv_json,
        mdp_graph,
        this_file_metadata,
        full_file_name,
        policy_player,
        Q_ITERATIONS=Q_ITERATIONS,
        DISCOUNT_FACTOR=DISCOUNT_FACTOR,
        Q_ALPHA=Q_ALPHA,
        Q_EPSILON=Q_EPSILON,
        IRL_SOLVER_EPSILON=IRL_SOLVER_EPSILON,
        policy_player_max_step=data_length,
        verbose=VERBOSE,
        irl_solver_iterations=IRL_SOLVER_ITERATIONS,
        q_learning_episode_length=q_learning_episode_length
    )

    compare_processor = CompareProcessor()
    compare_processor.compare(
        irl_result,
        full_file_name,
        conv_json,
        this_file_metadata,
        transition_frame_step,
        mdp_graph.Ca.shape,
        policy_player,
        policy_player_max_step=data_length,
        show_plot=VERBOSE,max_time_frames=maximum_time_size_frames,
        heatmap_folder_path=heatmap_folder_path

    )

    saver_plotter = IrlResultsPlotterSaver(
        filename,
        irl_result.list_of_t_W_intercept_policies_rewards,
        IRL_SOLVER_ITERATIONS
    )

    saver_plotter.plot(policies_save_folder_path)
