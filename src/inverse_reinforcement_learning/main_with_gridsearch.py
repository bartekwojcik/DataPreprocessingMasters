from settings import Settings
import mdp_const
from inverse_reinforcement_learning.main import main_async
import asyncio

if __name__ == "__main__":
    VERBOSE = False
    max_time_steps = [7, 0.5, 1]
    discount_factors = [0.7, 0.99, 0.90]
    policies_theta = [0.005, 0.01, 0.001]



    for ts in max_time_steps:
        for gamma in discount_factors:
            for theta in policies_theta:
                global_prefix = (
                    f"time_step_{ts}_gamma_{gamma}_theta_{theta}"
                )

                settings = Settings(ts,gamma,theta,0.1,global_prefix)

                asyncio.run(main_async(settings, VERBOSE))
