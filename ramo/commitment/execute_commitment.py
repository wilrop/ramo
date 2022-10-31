from collections import defaultdict

import numpy as np

from ramo.game.properties import get_player_actions, get_num_objectives, get_num_players
from ramo.utils.agent_loader import create_agents
from ramo.utils.experiments import calc_com_probs, calc_returns, calc_action_probs, get_payoffs


def get_leader(agents, episode, alternate=False):
    """Select the leader in the current episode.

    Args:
        agents (List[Agent]): A list of agents.
        episode (int): The current episode.
        alternate (bool, optional): Whether to alternate the leader between players of not. (Default value = False)

    Returns:
        Tuple[int, Agent]: The id of the leader and the leader agent itself.

    """
    if alternate:
        leader = episode % len(agents)
    else:
        leader = 0

    for id, agent in enumerate(agents):
        if id == leader:
            agent.make_leader()
        else:
            agent.make_follower()

    return leader, agents[leader]


def select_actions(agents, commitment):
    """Select an action from each agent's policy.

    Args:
        agents (List[Agent]): A list of agents.
        commitment (None | int | ndarray): The commitment from the leader.

    Returns:
        List[int]: A list of selected actions.

    """
    selected = []
    for agent in agents:
        selected.append(agent.select_action(commitment))
    return selected


def update(agents, commitment, actions, payoffs):
    """Perform an update for a list of agents.

    Args:
        agents (List[Agent]): A list of agents.
        commitment (int | ndarray): The commitment from the leader.
        actions (List[int]): A list of each action that was chosen, indexed by agent.
        payoffs (List[ndarray]): A list of each payoff that was received, indexed by agent.

    Returns:

    """
    for agent, payoff in zip(agents, payoffs):
        agent.update(commitment, actions, payoff)


def execute_commitment(payoff_matrices, u_tpl, experiment='coop_action', runs=100, episodes=5000, rollouts=100,
                       alternate=False, alpha_lq=0.01, alpha_ltheta=0.01, alpha_fq=0.01, alpha_ftheta=0.01,
                       alpha_cq=0.01, alpha_ctheta=0.01, alpha_q_decay=1, alpha_theta_decay=1, alpha_com_decay=1,
                       seed=None):
    """Execute a commitment experiment.

    Args:
        payoff_matrices (List[ndarray]): A list of payoff matrices representing the MONFG.
        u_tpl (Tuple[callable]): A tuple of utility functions.
        experiment (str, optional): The type of commitment experiment to execute. (Default value = 'coop_action')
        runs (int, optional): The number of times to repeat the experiment. (Default value = 100)
        episodes (int, optional): The number of episodes in one run of the experiment. (Default value = 5000)
        rollouts (int, optional): The number of Monte-Carlo simulations at each episode. (Default value = 100)
        alternate (bool, optional): Whether to alternate the players between leader and follower.
            (Default value = False)
        alpha_lq (float, optional): The learning rate for leader Q-values. (Default value = 0.01)
        alpha_ltheta (float, optional): The learning rate for leader policy parameters. (Default value = 0.01)
        alpha_fq (float, optional): The learning rate for follower Q-values. (Default value = 0.01)
        alpha_ftheta (float, optional): The learning rate for follower policy parameters. (Default value = 0.01)
        alpha_cq (float, optional): The learning rate for optional commitment Q-values. (Default value = 0.01)
        alpha_ctheta (float, optional): The learning rate for optional commitment policy parameters.
            (Default value = 0.01)
        alpha_q_decay (float, optional): The decay for the Q-values learning rate. (Default value = 1)
        alpha_theta_decay (float, optional): The decay for the policy parameters learning rate. (Default value = 1)
        alpha_com_decay (float, optional): The decay for the commitment strategy learning rate when using optional
            commitment. (Default value = 1)
        seed (int, optional): The seed for random number generation. (Default value = None)

    Returns:
        List[Agent]: A list of trained agents.

    Raises:
        Exception: When the number of players does not equal two.

    """
    num_agents = get_num_players(payoff_matrices)

    if num_agents != 2:
        raise Exception(f'Commitment experiments with {num_agents} are currently not supported')

    rng = np.random.default_rng(seed=seed)  # Set the seed.

    # Some basic setup.
    player_actions = get_player_actions(payoff_matrices)
    num_objectives = get_num_objectives(payoff_matrices)

    returns_log = defaultdict(list)
    action_probs_log = defaultdict(list)
    com_probs_log = defaultdict(list)
    joint_action_log = []
    metadata = {
        'payoff_matrices': list(map(lambda x: x.tolist(), payoff_matrices)),
        'u_tpl': u_tpl,
        'experiment': experiment,
        'runs': runs,
        'episodes': episodes,
        'rollouts': rollouts,
        'alternate': alternate,
        'alpha_lq': alpha_lq,
        'alpha_ltheta': alpha_ltheta,
        'alpha_fq': alpha_fq,
        'alpha_ftheta': alpha_ftheta,
        'alpha_cq': alpha_cq,
        'alpha_ctheta': alpha_ctheta,
        'alpha_q_decay': alpha_q_decay,
        'alpha_theta_decay': alpha_theta_decay,
        'alpha_com_decay': alpha_com_decay,
        'seed': seed
    }

    for run in range(runs):
        print("Starting run: ", run)
        agents = create_agents(experiment, u_tpl, num_agents, player_actions, num_objectives, alpha_q=alpha_lq,
                               alpha_theta=alpha_ltheta, alpha_fq=alpha_fq, alpha_ftheta=alpha_ftheta,
                               alpha_cq=alpha_cq, alpha_ctheta=alpha_ctheta, alpha_q_decay=alpha_q_decay,
                               alpha_theta_decay=alpha_theta_decay, alpha_com_decay=alpha_com_decay, rng=rng)

        for episode in range(episodes):
            # We keep the actions and payoffs of this episode so that we can later calculate the SER.
            ep_actions = defaultdict(list)
            ep_payoffs = defaultdict(list)
            ep_commitments = []

            leader_id, leader = get_leader(agents, episode, alternate=alternate)

            for rollout in range(rollouts):  # Required to evaluate the SER and action probabilities.
                commitment = leader.get_commitment()
                actions = select_actions(agents, commitment)
                payoffs = get_payoffs(actions, payoff_matrices)

                # Log the results of this roll
                for idx in range(num_agents):
                    ep_actions[idx].append(actions[idx])
                    ep_payoffs[idx].append(payoffs[idx])
                ep_commitments.append(commitment)

            # Update the agent after the episode.
            # We use the last action and payoff to update the agent.
            last_actions = np.array([ep_actions[ag][-1] for ag in range(num_agents)])
            last_payoffs = np.array([ep_payoffs[ag][-1] for ag in range(num_agents)])
            last_commitment = ep_commitments[-1]
            update(agents, last_commitment, last_actions, last_payoffs)  # Update the agents.

            # Get the necessary results from this episode.
            action_probs = calc_action_probs(ep_actions, player_actions, rollouts)
            returns = calc_returns(ep_payoffs, agents, rollouts)
            com_probs = calc_com_probs(ep_commitments, rollouts)

            # Append the logs.
            for idx in range(num_agents):
                returns_log[idx].append([run, episode, returns[idx]])
                prob_log = [run, episode] + action_probs[idx].tolist()
                action_probs_log[idx].append(prob_log)
            com_log = [run, episode] + com_probs
            com_probs_log[leader_id].append(com_log)

            joint_action_log.append([run, episode] + last_actions.tolist())

    return returns_log, action_probs_log, joint_action_log, com_probs_log, metadata
