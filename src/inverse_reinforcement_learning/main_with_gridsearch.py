from settings import Settings
import mdp_const
from inverse_reinforcement_learning.main import main_async
import asyncio

if __name__ == "__main__":
    VERBOSE = False

    q_iterations = [1000,2000]
    q_epsilon = [0.05,0.1,0.2]

    for iterations in q_iterations:
        for epislon in q_epsilon:

            global_prefix = (
                f"QITERS_{iterations}_QEPSILON_{q_epsilon}"
            )
            settings = Settings(MAX_CONTINUOUS_TIME_SEC=10.0,
                                DISCOUNT_FACTOR=0.99999999,
                                POLICY_THETA=0.001,
                                IRL_SOLVER_EPSILON=0.05,
                                Q_ITERATIONS=900,
                                Q_ALPHA=0.5,
                                Q_EPSILON=0.25,
                                GLOBAL_PREFIX_FOR_FILE_NAMES= global_prefix
                                )

            asyncio.run(main_async(settings, VERBOSE))
