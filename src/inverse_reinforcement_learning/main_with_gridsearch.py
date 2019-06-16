from settings import Settings
import mdp_const
from inverse_reinforcement_learning.main import main_async
import asyncio

if __name__ == "__main__":
    VERBOSE = False
    max_time_steps = [0.5, 1]
    discount_factors = [0.99,]
    policies_theta = [0.0001]



    for ts in max_time_steps:
        for gamma in discount_factors:
            for theta in policies_theta:

                #TODO THIS IS NOT GOOD ENOUGH FOR Q LEARNING

                global_prefix = (
                    f"time_step_{ts}_gamma_{gamma}_theta_{theta}"
                )
                settings = Settings(MAX_CONTINUOUS_TIME_SEC=ts,
                                    DISCOUNT_FACTOR=gamma,
                                    POLICY_THETA=theta,
                                    IRL_SOLVER_EPSILON=0.05,
                                    Q_ITERATIONS=900,
                                    Q_ALPHA=0.5,
                                    Q_EPSILON=0.25,
                                    GLOBAL_PREFIX_FOR_FILE_NAMES= global_prefix
                                    )
                # TODO THIS IS NOT GOOD ENOUGH FOR Q LEARNING
                asyncio.run(main_async(settings, VERBOSE))
