from settings import Settings
import os
import json

settings = Settings(
    MAX_CONTINUOUS_TIME_SEC=10.0,
    DISCOUNT_FACTOR=0.999999,
    POLICY_THETA=0.01,
    IRL_SOLVER_EPSILON=0.05,
    Q_ITERATIONS=100,
    Q_ALPHA=0.4,
    Q_EPSILON=0.2,
)



master_conversation = []
METADATA_PATH = settings.READABLE_METADATA_FILE_PATH
with open(METADATA_PATH, "r") as metadata_file:
    metadata_json = json.loads(metadata_file.read())

    for file in os.listdir(settings.HUMAN_READABLE_FOLDER_PATH):
        file_path = os.path.join(settings.HUMAN_READABLE_FOLDER_PATH,file)

        if file == "human_readable_conversation_99.json":
            continue

        with open(file_path, "r") as conv_file:
            print(file_path)
            conversation = json.load(conv_file)
            master_conversation.extend(conversation)
            debug = 5

save_path = os.path.join(settings.HUMAN_READABLE_FOLDER_PATH, "human_readable_conversation_99.json")
with open(save_path, "w") as new_file:
    json.dump(master_conversation, new_file)
