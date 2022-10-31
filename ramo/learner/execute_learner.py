from collections import defaultdict

import numpy as np

from ramo.game.properties import get_player_actions, get_num_objectives, get_num_players
from ramo.utils.agent_loader import create_agents
from ramo.utils.experiments import calc_returns, calc_action_probs, get_payoffs


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


def update(agents, actions, payoffs, experiment):
    """Perform an update for a list of agents.

    Args:
        agents (List[Agent]): A list of agents.
        actions (List[int]): A list of each action that was chosen, indexed by agent.
        payoffs (List[ndarray]): A list of each payoff that was received, indexed by agent.

    Returns:

    """
    if experiment.startswith('indep'):
        for agent, action, payoff in zip(agents, actions, payoffs):
            agent.update(action, payoff)
    else:
        for agent, payoff in zip(agents, payoffs):
            agent.update(actions, payoff)


def execute_learner(payoff_matrices, u_tpl, experiment='indep_ac', runs=100, episodes=5000, rollouts=100,
                    alpha_q=0.01, alpha_theta=0.01, alpha_q_decay=1, alpha_theta_decay=1, epsilon=1,
                    epsilon_decay=0.995, min_epsilon=0.1, seed=None):
    """Execute a commitment experiment.

    Args:
        payoff_matrices (List[ndarray]): A list of payoff matrices representing the MONFG.
        u_tpl (Tuple[callable]): A tuple of utility functions.
        experiment (str, optional): The type of commitment experiment to execute. (Default value = 'coop_action')
        runs (int, optional): The number of times to repeat the experiment. (Default value = 100)
        episodes (int, optional): The number of episodes in one run of the experiment. (Default value = 5000)
        rollouts (int, optional): The number of Monte-Carlo simulations at each episode. (Default value = 100)
        alpha_q (float, optional): The learning rate for Q-values. (Default value = 0.01)
        alpha_theta (float, optional): The learning rate for policy parameters. (Default value = 0.01)
        alpha_q_decay (float, optional): The decay of the learning rate for Q-values. (Default value = 1)
        alpha_theta_decay (float, optional): The decay for the learning rate of policy parameters. (Default value = 1)
        epsilon (float, optional): The exploration rate for a Q-learner agent. (Default value = 1)
        epsilon_decay (float, optional): The decay for the exploration rate. (Default value = 0.995)
        min_epsilon (float, optional): The minimum value for the exploration rate. (Default value = 0.1)
        seed (int, optional): The seed for random number generation. (Default value = None)

    Returns:
        Tuple[Dict, Dict, ndarray, Dict]: A log of payoffs, a log of action probabilities for both agents, a log of the
            state distribution and a log of the commitment probabilities.

    Raises:
        Exception: When the number of players does not equal two.

    """
    rng = np.random.default_rng(seed=seed)  # Initialise a random number generator.

    player_actions = get_player_actions(payoff_matrices)
    num_objectives = get_num_objectives(payoff_matrices)
    num_agents = get_num_players(payoff_matrices)

    # Set up logging data structures.
    returns_log = defaultdict(list)
    action_probs_log = defaultdict(list)
    joint_action_log = []
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

    for run in range(runs):
        print("Starting run: ", run)
        agents = create_agents(experiment, u_tpl, num_agents, player_actions, num_objectives, alpha_q=alpha_q,
                               alpha_theta=alpha_theta, alpha_q_decay=alpha_q_decay,
                               alpha_theta_decay=alpha_theta_decay, epsilon=epsilon, epsilon_decay=epsilon_decay,
                               min_epsilon=min_epsilon, rng=rng)

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
            update(agents, last_actions, last_payoffs, experiment)  # Update the agents.

            # Get the necessary results from this episode.
            action_probs = calc_action_probs(ep_actions, player_actions, rollouts)
            returns = calc_returns(ep_payoffs, agents, rollouts)

            # Append the logs.
            for idx in range(num_agents):
                returns_log[idx].append([run, episode, returns[idx]])
                prob_log = [run, episode] + action_probs[idx].tolist()
                action_probs_log[idx].append(prob_log)

            joint_action_log.append([run, episode] + last_actions.tolist())

    return returns_log, action_probs_log, joint_action_log, metadata
