import pandas as pd


def save_data(path, name, game, num_agents, player_actions, runs, episodes, returns_log=None, action_probs_log=None,
              state_dist_log=None, com_probs_log=None):
    """Save results from an experiment to disk in CSV format.

    Args:
      path (str): The path to the directory in which all files will be saved.
      name (str): The name of the experiment.
      game (str): The name of the game that was played.
      num_agents (int): The number of agents in the experiment.
      player_actions (Tuple[int]): A tuple with the number of actions per agent.
      runs (int): The number of runs the experiment was executed.
      episodes (int): The number of episodes in each run.
      returns_log (dict, optional): The log for the returns. (Default = None)
      action_probs_log (dict, optional): The log for the action probabilities. (Default = None)
      state_dist_log (ndarray, optional): The state distribution log. (Default = None)
      com_probs_log (dict, optional): The log for the commitment probabilities. (Default = None)

    Returns:

    """
    default_cols = ['Trial', 'Episode']
    optional_commitment_experiments = ['opt_comp_action', 'opt_coop_action', 'opt_coop_policy']

    if returns_log is not None:
        returns_columns = default_cols + ['Payoff']
        for idx in range(num_agents):
            df_r = pd.DataFrame(returns_log[idx], columns=returns_columns)
            df_r.to_csv(f'{path}/{name}_{game}_A{idx + 1}_returns.csv', index=False)

    if action_probs_log is not None:
        for idx in range(num_agents):
            action_columns = default_cols + [f'Action {a + 1}' for a in range(player_actions[idx])]
            df_a = pd.DataFrame(action_probs_log[idx], columns=action_columns)
            df_a.to_csv(f'{path}/{name}_{game}_A{idx + 1}_probs.csv', index=False)

    if state_dist_log is not None:
        state_dist_log /= runs * (0.1 * episodes)
        df_s = pd.DataFrame(state_dist_log)
        df_s.to_csv(f'{path}/{name}_{game}_states.csv', index=False, header=False)

    if com_probs_log is not None and name in optional_commitment_experiments:
        com_columns = default_cols + ['Communication', 'No communication']
        for idx in range(num_agents):
            df_c = pd.DataFrame(com_probs_log[idx], columns=com_columns)
            df_c.to_csv(f'{path}/{name}_{game}_A{idx + 1}_com.csv', index=False)
