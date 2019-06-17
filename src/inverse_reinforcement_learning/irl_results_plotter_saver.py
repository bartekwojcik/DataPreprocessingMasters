from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import os
import json

from settings import Settings


class IrlResultsPlotterSaver:
    """
    Saves weights and intercept to numpy file AND also plots t over time
    I know, i am just in hurry
    """

    def __init__(
            self,
            file_name:str,
            list_of_ts_weights_intercept:List[Tuple[float, np.ndarray, np.ndarray,np.ndarray,np.ndarray]],
            iterations:int,
            settings:Settings):

        self.file_name = f"{settings.GLOBAL_PREFIX_FOR_FILE_NAMES}_{file_name}"
        self.settings = settings
        self.iterations = iterations
        self.list_of_ts_weights_intercept_policy_reward = list_of_ts_weights_intercept

    def plot(self):
        """
        Eh
        :return:
        """

        folder_name = f"frame_{self.settings.TRANSITION_FRAME_STEP}" \
                      f"_QITERS_{self.settings.Q_ITERATIONS}" \
                      f"_QEPSILON_{self.settings.Q_EPSILON}"
        folder_path = os.path.join(
            self.settings.COMPARISON_PLOTS_FOLDER_PATH, folder_name
        )
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)



        length = len(self.list_of_ts_weights_intercept_policy_reward)
        t_values = np.zeros((length,))
        w_diff_values = np.zeros((length-1,))
        policies_diff_values = np.zeros((length-1,))
        rewards_diff_values = np.zeros((length-1,))

        t, W, intercept, policy, reward = self.list_of_ts_weights_intercept_policy_reward[0]

        w_full_values = np.zeros(((length,)+W.shape))
        qs_values = []
        policies_full_values = np.zeros(((length,)+policy.shape))
        rewards_full_values = np.zeros(((length,)+reward.shape))

        n_i = len(self.list_of_ts_weights_intercept_policy_reward)
        iterations = np.arange(0,n_i,1)

        for i,(t,W,Q,policy,reward) in enumerate(self.list_of_ts_weights_intercept_policy_reward):
            t_values[i] = t
            w_full_values[i] = W
            qs_values.append(dict(Q))
            policies_full_values[i] = policy
            rewards_full_values[i] = reward


            if i == 0:
                continue
            else:
                t_previous,\
                W_previous,\
                Q_previous, \
                policy_previous, \
                reward_previous = self.list_of_ts_weights_intercept_policy_reward[i-1]

                w_intercept = W
                previous_w_intercept = W_previous
                w_diff = np.linalg.norm(
                    np.asarray(w_intercept)
                    - np.asarray(previous_w_intercept)
                )
                w_diff_values[i-1] =w_diff

                policy_diff = np.linalg.norm(
                    np.asarray(policy)
                    - np.asarray(policy_previous)
                )
                policies_diff_values[i-1] =policy_diff

                reward_diff = np.linalg.norm(
                    np.asarray(reward)
                    - np.asarray(reward_previous)
                )
                rewards_diff_values[i-1] =reward_diff

        title = f"{self.file_name} t value over iterations"
        self.__save_plot(folder_path,
                         self.file_name,
                         "T_VALUES_plot",
                         t_values, iterations, title,
                         "iteration",
                         "value of t")


        self.__save_numpy_to_file(folder_path, f"{self.file_name}_w_values", w_full_values)
        self.__save_numpy_to_file(folder_path, f"{self.file_name}_Q_values", qs_values)
        self.__save_numpy_to_file(folder_path, f"{self.file_name}_policies", policies_full_values)
        self.__save_numpy_to_file(folder_path, f"{self.file_name}_rewards", rewards_full_values)
        self.__save_numpy_to_file(folder_path, f"{self.file_name}_T_values", t_values)

        if len(iterations) > 1:
            title = f"{self.file_name} difference of w+intercept per iteration (minus w previous + intercept previous)"
            shorter_iterations = iterations[1:]
            self.__save_plot(folder_path,
                             self.file_name,
                             "W_difference",
                             w_diff_values, shorter_iterations, title,
                             "iteration",
                             "W - previous W value")

            title = f"{self.file_name} policy - previous policy"
            self.__save_plot(folder_path,
                             self.file_name,
                             "policy_difference",
                             policies_diff_values, shorter_iterations, title,
                             "iteration",
                             "policy - previous policy value")

            title = f"{self.file_name} reward - previous reward"
            self.__save_plot(folder_path,
                             self.file_name,
                             "rewards_difference",
                             rewards_diff_values, shorter_iterations, title,
                             "iteration",
                             "reward - previous reward value")




    def __save_plot(self,folder_path, file_name,plot_file_name,values, iterations,title:str, x_label, y_label):
        """
        plots values per iteration
        :param folder_path: 
        :param file_name: 
        :param values: 
        :param iterations: 
        :param title: 
        :param x_label: 
        :param y_label: 
        :return: 
        """

        file_path = os.path.join(
            self.settings.COMPARISON_PLOTS_FOLDER_PATH, folder_path,
            f"{self.settings.GLOBAL_PREFIX_FOR_FILE_NAMES}_{file_name}_{plot_file_name}.png",
        )


        fig = plt.figure()
        fig.suptitle(
            title, fontsize=14, fontweight="bold"
        )
        plt.plot(iterations,values)
        plt.ylabel(y_label)#")
        plt.xlabel(x_label)#")
        fig.set_size_inches((18, 9), forward=False)
        plt.savefig(file_path, quality=70, dpi= 400)
        plt.close(fig)

    def __save_numpy_to_file(self, folder_path, file_name, values):

        full_path = os.path.join(folder_path,file_name )
        np.save(full_path, values)




