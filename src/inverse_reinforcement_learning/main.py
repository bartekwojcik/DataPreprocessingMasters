import multiprocessing

import numpy as np
import settings
import os
import json
from inverse_reinforcement_learning.process_file import process_file, async_process_file


if __name__ == "__main__":
    HUMAN_READABLE_FOLDER_PATH = settings.HUMAN_READABLE_FOLDER_PATH
    METADATA_PATH = settings.READABLE_METADATA_FILE_PATH
    VERBOSE = True

    with open(METADATA_PATH, "r") as metadata_file:

        metadata_json = json.loads(metadata_file.read())

        for filename in os.listdir(HUMAN_READABLE_FOLDER_PATH):
            full_file_name = os.path.join(HUMAN_READABLE_FOLDER_PATH, filename)

            with open(full_file_name, "r") as conversation_file:

                process_file(metadata_json,filename,conversation_file,full_file_name,VERBOSE)
                #async_process_file(metadata_json,filename,conversation_file,full_file_name,VERBOSE)


