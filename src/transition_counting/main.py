import os
import settings
import json
import numpy as np
from transition_counting.transition_counter import TransitionCounter

if __name__ == '__main__':

    folder_path = settings.HUMAN_READABLE_FOLDER_PATH
    result = np.zeros((2,2,2,2))
    counter = TransitionCounter()
    for filename in os.listdir(folder_path):
        full_file_name = os.path.join(folder_path,filename)
        with open(full_file_name, 'r') as data:
            #TODO start at some pouit between 0 and 1 second, and get through all data, dont miss anything :/
            data = json.loads(data.read())
            file_result = counter.count_transitions(data,20)
            result += file_result

    print(result)
