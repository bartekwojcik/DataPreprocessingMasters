from settings import Settings
import mdp_const
from inverse_reinforcement_learning.main import main_async
import asyncio


def do_grid():
    print("STARTED")

    VERBOSE = False

    q_iterations = [1000,2000]
    q_epsilon = [0.05,0.1,0.2]

    for iterations in q_iterations:
        for epsilon in q_epsilon:

            print(f"started iterations: {iterations} epsilon: {epsilon}")

            global_prefix = (
                f"QITERS_{iterations}_QEPSILON_{epsilon}"
            )
            settings = Settings(MAX_CONTINUOUS_TIME_SEC=10.0,
                                DISCOUNT_FACTOR=0.99999999,
                                POLICY_THETA=0.001,
                                IRL_SOLVER_EPSILON=0.05,
                                Q_ITERATIONS=iterations,
                                Q_ALPHA=0.5,
                                Q_EPSILON=epsilon,
                                GLOBAL_PREFIX_FOR_FILE_NAMES= global_prefix
                                )

            loop = asyncio.get_event_loop()
            loop.run_until_complete(main_async(settings, VERBOSE))
            loop.close()

    print("ENDED")

# if __name__ == "__main__":
#     do_grid()