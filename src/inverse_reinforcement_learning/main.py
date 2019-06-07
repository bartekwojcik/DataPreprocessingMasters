import multiprocessing
import threading

import numpy as np
from settings import Settings
import os
import json
from inverse_reinforcement_learning.process_file import process_file, async_process_file
import asyncio


async def main_async(settings:Settings, VERBOSE:bool):
    HUMAN_READABLE_FOLDER_PATH = settings.HUMAN_READABLE_FOLDER_PATH
    METADATA_PATH = settings.READABLE_METADATA_FILE_PATH


    tasks = []
    loop = asyncio.get_event_loop()
    with open(METADATA_PATH, "r") as metadata_file:
        metadata_json = json.loads(metadata_file.read())

        for filename in os.listdir(HUMAN_READABLE_FOLDER_PATH):
            full_file_name = os.path.join(HUMAN_READABLE_FOLDER_PATH, filename)

            with open(full_file_name, "r") as conversation_file:
                conv_json = json.loads(conversation_file.read())
                task = async_process_file(loop, metadata_json, filename, conv_json, full_file_name, VERBOSE, settings)
                tasks.append(task)

    await asyncio.gather(*(tasks)[3:4])


if __name__ == "__main__":
    VERBOSE = True
    settings = Settings(1, 0.99, 0.001, 0.1)
    asyncio.run(main_async(settings, VERBOSE))
