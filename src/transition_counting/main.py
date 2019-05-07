import os
import settings
import json
import numpy as np
from transition_counting.transition_counter import TransitionCounter
from transition_counting.gaze_processor import GazeProcessor

if __name__ == "__main__":

    folder_path = settings.HUMAN_READABLE_FOLDER_PATH
    result = np.zeros((2, 2, 2, 2))
    counter = TransitionCounter()

    for filename in os.listdir(folder_path):
        full_file_name = os.path.join(folder_path, filename)

        with open(full_file_name, "r") as data_raw:
            frame_step = 12
            starting_points = np.arange(0, frame_step)
            data = json.loads(data_raw.read())

            for i in starting_points:
                file_result = counter.count_transitions(data, frame_step, i)
                result += file_result

            file_end_debug = 5
    gaze_processor = GazeProcessor()
    print(result)
    result_file_path = os.path.join(
        settings.MY_DATA_FOLDER_PATH, "transition_counting_results"
    )
    np.save(result_file_path, result)

    # OTHER IN-> OUT
    print("OTHER IN-> OUT #############")
    in_out_in_in = gaze_processor.decode_matrix(result, 1, 0, 1, 1)
    print(f"in_out_in_in: {in_out_in_in }")
    in_out_in_out = gaze_processor.decode_matrix(result, 1, 0, 1, 0)
    print(f"in_out_in_out: {in_out_in_out }")
    in_out_out_out = gaze_processor.decode_matrix(result, 1, 0, 0, 0)
    print(f"in_out_out_out: {in_out_out_out }")
    in_out_out_in = gaze_processor.decode_matrix(result, 1, 0, 0, 1)
    print(f"in_out_out_in: {in_out_out_in }")

    # OTHER IN-> IN
    print("OTHER IN-> IN #############")
    in_in_in_in = gaze_processor.decode_matrix(result, 1, 1, 1, 1)
    print(f"in_in_in_in: {in_in_in_in }")
    in_in_in_out = gaze_processor.decode_matrix(result, 1, 1, 1, 0)
    print(f"in_in_in_out: {in_in_in_out }")
    in_in_out_out = gaze_processor.decode_matrix(result, 1, 1, 0, 0)
    print(f"in_in_out_out: {in_in_out_out }")
    in_in_out_in = gaze_processor.decode_matrix(result, 1, 1, 0, 1)
    print(f"in_in_out_in: {in_in_out_in }")

    # OTHER OUT-> OUT
    print("OTHER OUT-> OUT #############")
    out_out_in_in = gaze_processor.decode_matrix(result, 0, 0, 1, 1)
    print(f"out_out_in_in: {out_out_in_in }")
    out_out_in_out = gaze_processor.decode_matrix(result, 0, 0, 1, 0)
    print(f"out_out_in_out: {out_out_in_out }")
    out_out_out_out = gaze_processor.decode_matrix(result, 0, 0, 0, 0)
    print(f"out_out_out_out: {out_out_out_out }")
    out_out_out_in = gaze_processor.decode_matrix(result, 0, 0, 0, 1)
    print(f"out_out_out_in: {out_out_out_in }")

    # OTHER OUT-> IN
    print("OTHER OUT-> IN #############")
    out_in_in_in = gaze_processor.decode_matrix(result, 0, 1, 1, 1)
    print(f"out_in_in_in: {out_in_in_in }")
    out_in_in_out = gaze_processor.decode_matrix(result, 0, 1, 1, 0)
    print(f"out_in_in_out: {out_in_in_out }")
    out_in_out_out = gaze_processor.decode_matrix(result, 0, 1, 0, 0)
    print(f"out_in_out_out: {out_in_out_out }")
    out_in_out_in = gaze_processor.decode_matrix(result, 0, 1, 0, 1)
    print(f"out_in_out_in: {out_in_out_in }")
