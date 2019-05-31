import json

import settings
from Mdp.mdp_utils import MdpUtils
from inverse_reinforcement_learning.compare_processor import CompareProcessor
from inverse_reinforcement_learning.irl_processor import IrlProcessor
import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from inverse_reinforcement_learning.get_model_probas import ModelProbasGetter


async def async_process_file(loop ,metadata_json, filename, conv_json,full_file_name,VERBOSE):

     return await loop.run_in_executor(ProcessPoolExecutor(), process_file, metadata_json, filename, conv_json, full_file_name,VERBOSE)
    # process_file(metadata_json, filename, conv_json, full_file_name,VERBOSE)

    # async_func = asyncio.coroutine(process_file(metadata_json, filename, conv_json,full_file_name,VERBOSE))
    # return await async_func()

def process_file(metadata_json, filename, conv_json,full_file_name,VERBOSE):

    this_file_metadata = metadata_json[filename]

    probas_getter = ModelProbasGetter()
    mdp_graph = probas_getter.get_model_probas(conv_json,this_file_metadata,settings.TRANSITION_FRAME_STEP,filename,VERBOSE)

    #plots probabilities of model's actions in states
    # mdp_graph.plot_probabilities_per_state(VERBOSE,filename)

    data_length = len(conv_json)

    processor = IrlProcessor()
    irl_result = processor.process(
        conv_json,
        mdp_graph,
        this_file_metadata,
        full_file_name,
        policy_player_max_step=data_length,
        verbose=VERBOSE,
    )

    compare_processor = CompareProcessor()
    compare_processor.compare(
        irl_result,
        full_file_name,
        conv_json,
        this_file_metadata,
        settings.TRANSITION_FRAME_STEP,
        mdp_graph.Ca.shape,
        show_plot=VERBOSE,
    )

    print(
        f"!!!!!!!!!!!!!!!!!!!!file {filename} done#################################"
    )
    # TODO might do something with irl_result later ¯\_(ツ)_/¯ asd