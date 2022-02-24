import argparse
import time
from collections import defaultdict

import numpy as np

import mo_gt.games.games as games
from mo_gt.utils.data import save_metadata, save_data
from mo_gt.utils.experiments import create_game_path, calc_returns, calc_action_probs, get_payoffs, create_agents


def select_actions(agents):
    """Select an action from each agent's policy.

    Args:
      agents (List[Agent]): A list of agents.

    Returns:
      List[int]: A list of selected actions.

    """
    selected = []
    for agent in agents:
        selected.append(agent.select_action())
    return selected


def update(agents, actions, payoffs):
    """Perform an update for a list of agents.

    Args:
      agents (List[Agent]): A list of agents.
      actions (List[int]): A list of each action that was chosen, indexed by agent.
      payoffs (List[ndarray]): A list of each payoff that was received, indexed by agent.

    Returns:

    """
    for agent, payoff in zip(agents, payoffs):
        agent.update(actions, payoff)


def execute_learner(payoff_matrices, u_tpl, experiment='coop_action', runs=100, episodes=5000, rollouts=100,
                    alpha_q=0.01, alpha_theta=0.01, alpha_q_decay=1, alpha_theta_decay=1, epsilon=1,
                    epsilon_decay=0.995, min_epsilon=0.1, seed=1):
    """Execute a commitment experiment.

    Args:
        payoff_matrices (List[ndarray]): A list of payoff matrices representing the MONFG.
        u_tpl (Tuple[callable]): A tuple of utility functions.
        experiment (str, optional): The type of commitment experiment to execute. (Default = 'coop_action')
        runs (int, optional): The number of times to repeat the experiment. (Default = 100)
        episodes (int, optional): The number of episodes in one run of the experiment. (Default = 100)
        rollouts (int, optional): The number of Monte-Carlo simulations at each episode. (Default = 100)
        alpha_q (float, optional): The learning rate for Q-values. (Default = 0.2)
        alpha_theta (float, optional): The learning rate for policy parameters. (Default = 0.005)
        alpha_q_decay (float, optional): The decay of the learning rate for Q-values. (Default = 1)
        alpha_theta_decay (float, optional): The decay for the learning rate of policy parameters. (Default = 1)
        epsilon (float, optional): The exploration rate for a Q-learner agent. (Default = 1)
        epsilon_decay (float, optional): The decay for the exploration rate. (Default = 0.995)
        min_epsilon (float, optional): The minimum value for the exploration rate. (Default = 0.1)
        seed (int, optional): The seed for random number generation. (Default = 1)

    Returns:
        Tuple[Dict, Dict, ndarray, Dict}: A log of payoffs, a log of action probabilities for both agents, a log of the
            state distribution and a log of the commitment probabilities.

    Raises:
        Exception: When the number of players does not equal two.

    """
    np.random.seed(seed=seed)

    player_actions = payoff_matrices[0].shape[:-1]
    num_objectives = payoff_matrices[0].shape[-1]

    # Set up logging data structures.
    returns_log = defaultdict(list)
    action_probs_log = defaultdict(list)
    state_dist_log = np.zeros(player_actions)
    metadata = {
        'payoff_matrices': list(map(lambda x: x.tolist(), payoff_matrices)),
        'u_tpl': u_tpl,
        'experiment': experiment,
        'runs': runs,
        'episodes': episodes,
        'rollouts': rollouts,
        'alpha_q': alpha_q,
        'alpha_theta': alpha_theta,
        'alpha_q_decay': alpha_q_decay,
        'alpha_theta_decay': alpha_theta_decay,
        'epsilon': epsilon,
        'epsilon_decay': epsilon_decay,
        'min_epsilon': min_epsilon,
        'seed': seed
    }

    start = time.time()

    for run in range(runs):
        print("Starting run: ", run)
        agents = create_agents(experiment, u_tpl, num_agents, player_actions, num_objectives, alpha_q=alpha_q,
                               alpha_theta=alpha_theta, alpha_q_decay=alpha_q_decay,
                               alpha_theta_decay=alpha_theta_decay, epsilon=epsilon, epsilon_decay=epsilon_decay,
                               min_epsilon=min_epsilon)

        for episode in range(episodes):
            # We keep the actions and payoffs of this episode so that we can later calculate the SER.
            ep_actions = defaultdict(list)
            ep_payoffs = defaultdict(list)

            for rollout in range(rollouts):  # Required to evaluate the SER and action probabilities.
                actions = select_actions(agents)
                payoffs = get_payoffs(actions, payoff_matrices)

                # Log the results of this roll
                for idx in range(num_agents):
                    ep_actions[idx].append(actions[idx])
                    ep_payoffs[idx].append(payoffs[idx])

            # Update the agent after the episode.
            # We use the last action and payoff to update the agent.
            last_actions = np.array([ep_actions[ag][-1] for ag in range(num_agents)])
            last_payoffs = np.array([ep_payoffs[ag][-1] for ag in range(num_agents)])
            update(agents, last_actions, last_payoffs)  # Update the agents.

            # Get the necessary results from this episode.
            action_probs = calc_action_probs(ep_actions, player_actions, rollouts)
            returns = calc_returns(ep_payoffs, agents, rollouts)

            # Append the logs.
            for idx in range(num_agents):
                returns_log[idx].append([run, episode, returns[idx]])
                prob_log = [run, episode] + action_probs[idx].tolist()
                action_probs_log[idx].append(prob_log)

            # If we are in the last 10% of episodes we build up a state distribution log.
            # This code is specific to two player games.
            if episode >= 0.9 * episodes:
                state_dist = np.zeros(player_actions)
                for a1, a2 in zip(ep_actions[0], ep_actions[1]):
                    state_dist[a1, a2] += 1
                state_dist /= rollouts
                state_dist_log += state_dist

    end = time.time()
    elapsed_mins = (end - start) / 60.0
    print("Minutes elapsed: " + str(elapsed_mins))

    return returns_log, action_probs_log, state_dist_log, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--game', type=str, default='game1', help="which MONFG game to play")
    parser.add_argument('--u', type=str, default=['u1', 'u2'], nargs='+',
                        help="Which utility functions to use per player")
    parser.add_argument('--experiment', type=str, default='indep_ac', help='The experiment to run.')
    parser.add_argument('--runs', type=int, default=100, help="number of trials")
    parser.add_argument('--episodes', type=int, default=5000, help="number of episodes")
    parser.add_argument('--rollouts', type=int, default=100, help="Rollout period for the policies")
    parser.add_argument('--dir', type=str, default='/Users/willemropke/Documents/mo-game-theory',
                        help='Parent directory for data and plots.')

    args = parser.parse_args()

    # Extracting the arguments.
    game = args.game
    u = args.u
    experiment = args.experiment
    runs = args.runs
    episodes = args.episodes
    rollouts = args.rollouts
    parent_dir = args.dir

    # Starting the experiments.
    payoff_matrices = games.get_monfg(game)
    data = execute_learner(payoff_matrices, u, experiment=experiment, runs=runs, episodes=episodes, rollouts=rollouts)
    returns_log, action_probs_log, state_dist_log, metadata = data

    # Writing the data to disk.
    num_agents = len(payoff_matrices)
    player_actions = tuple(payoff_matrices[0].shape[:-1])
    path = create_game_path('data', experiment, game, parent_dir=parent_dir)
    save_metadata(path, **metadata)
    save_data(path, experiment, game, num_agents, player_actions, runs, episodes, returns_log=returns_log,
              action_probs_log=action_probs_log, state_dist_log=state_dist_log)
