from settings import Settings
import mdp_const
from inverse_reinforcement_learning.main import main_async
import asyncio

if __name__ == "__main__":
    time_steps = [7, 0.5, 1]
    discount_factors = [0.7, 0.99, 0.90]
    policies_theta = [0.005, 0.01, 0.001]

    settings = Settings()

    for ts in time_steps:
        for gamma in discount_factors:
            for theta in policies_theta:

                settings.MAX_CONTINUOUS_TIME_SEC = ts
                settings.DISCOUNT_FACTOR = gamma
                settings.POLICY_THETA = theta
                settings.GLOBAL_PREFIX_FOR_FILE_NAMES = (
                    f"time_step_{ts}_gamma_{gamma}_theta_{theta}"
                )

                asyncio.run(main_async(settings))
