from concurrent.futures import ProcessPoolExecutor

from Mdp.at_high_model_components.at_high_policy_player import HighPolicyPlayer
from inverse_reinforcement_learning.compare_processor import CompareProcessor
from inverse_reinforcement_learning.get_model_probas import ModelProbasGetter
from inverse_reinforcement_learning.irl_processor import IrlProcessor
from inverse_reinforcement_learning.irl_results_plotter_saver import (
    IrlResultsPlotterSaver,
)
from settings import Settings


async def async_process_file(
    loop, metadata_json, filename, conv_json, full_file_name, VERBOSE, setting: Settings
):

    return await loop.run_in_executor(
        ProcessPoolExecutor(),
        process_file,
        metadata_json,
        filename,
        conv_json,
        full_file_name,
        VERBOSE,
        setting,
    )


def process_file(
    metadata_json, filename, conv_json, full_file_name, VERBOSE, settings: Settings
):
    print(f"file started: {full_file_name} ")

    this_file_metadata = metadata_json[filename]

    probas_getter = ModelProbasGetter()
    mdp_graph = probas_getter.get_model_probas(
        conv_json,
        this_file_metadata,
        settings.TRANSITION_FRAME_STEP,
        filename,
        VERBOSE,
        settings,
    )

    # plots probabilities of model's actions in states
    #mdp_graph.plot_probabilities_per_state(VERBOSE,filename, settings)

    data_length = len(conv_json)
    IRL_SOLVER_ITERATIONS = 5

    policy_player = HighPolicyPlayer(this_file_metadata, mdp_graph, 0.05)

    processor = IrlProcessor()
    irl_result = processor.process(
        conv_json,
        mdp_graph,
        this_file_metadata,
        full_file_name,
        policy_player,
        policy_player_max_step=data_length,
        verbose=VERBOSE,
        settings=settings,
        irl_solver_iterations=IRL_SOLVER_ITERATIONS,
    )

    compare_processor = CompareProcessor()
    compare_processor.compare(
        irl_result,
        full_file_name,
        conv_json,
        this_file_metadata,
        settings.TRANSITION_FRAME_STEP,
        mdp_graph.Ca.shape,
        settings,
        policy_player,
        policy_player_max_step=data_length,
        show_plot=VERBOSE,
    )

    saver_plotter = IrlResultsPlotterSaver(
        filename,
        irl_result.list_of_t_W_intercept_policies_rewards,
        IRL_SOLVER_ITERATIONS,
        settings,
    )

    saver_plotter.plot()
