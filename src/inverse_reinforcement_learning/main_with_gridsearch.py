import settings
from mdp_const import MdpConsts
from inverse_reinforcement_learning import main
import asyncio

if __name__ == '__main__':
 time_steps = [0.5,1]
 discount_factors = [0.99,0.90]
 policies_theta = [0.01, 0.001]

 for ts in time_steps:
     for gamma in discount_factors:
         for theta in policies_theta:
             MdpConsts.TIME_SIZE = ts
             settings.DISCOUNT_FACTOR = gamma
             settings.POLICY_THETA = theta
             settings.GLOBAL_PREFIX_FOR_FILE_NAMES = f"time_step_{ts}_gamma_{gamma}_theta_{theta}"
             asyncio.run(main.main())


