import argparse
import time
from collections import defaultdict

import numpy as np

import mo_gt.games.games as games
import mo_gt.games.utility_functions as uf
import mo_gt.utils.experiments as ex
from mo_gt.commitment.best_response_agent import BestResponseAgent
from mo_gt.commitment.comp_action_agent import CompActionAgent
from mo_gt.commitment.coop_action_agent import CoopActionAgent
from mo_gt.commitment.coop_policy_agent import CoopPolicyAgent
from mo_gt.commitment.non_stationary_agent import NonStationaryAgent
from mo_gt.commitment.optional_com_agent import OptionalComAgent
from mo_gt.learners.indep_actor_critic import IndependentActorCriticAgent
from mo_gt.utils.data import save_data


def get_leader(agents, episode, alternate=False):
    """Select the leader in the current episode.

    Args:
      agents (List[Agent]): A list of agents.
      episode (int): The current episode.
      alternate (bool, optional): Whether to alternate the leader between players of not. (Default = False)

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
      commitment: The commitment from the leader.

    Returns:
      List[int]: A list of selected actions.

    """
    selected = []
    for agent in agents:
        selected.append(agent.select_action(commitment))
    return selected


def get_payoffs(actions, payoff_matrices):
    """Get the payoffs from the payoff matrices from the selected actions.

    Args:
      actions (List[int]): The actions taken by each player.
      payoff_matrices (List[ndarray]): A list of payoff matrices.

    Returns:
      List[ndarray]: A list of received payoffs.

    """
    actions = tuple(actions)
    return list(map(lambda x: x[actions], payoff_matrices))


def calc_returns(payoffs_dict, agents, rollouts):
    """Calculate the scalarised expected returns for each agent.

    Args:
      payoffs_dict (Dict[int]: List[ndarray]): The vectorial payoffs obtained by the agents.
      agents (List[Agent]): A list of agents.
      rollouts (int): The amount of rollouts that were performed.

    Returns:
      List[float]: A list of scalarised expected returns.

    """
    returns = {}
    for ag, payoff_hist in payoffs_dict.items():
        payoff_sum = np.sum(payoff_hist, axis=0)
        avg_payoff = payoff_sum / rollouts
        ser = agents[ag].u(avg_payoff)
        returns[ag] = ser
    return returns


def calc_action_probs(actions_dict, player_actions, rollouts):
    """Calculate empirical action probabilities.

    Args:
      actions_dict (Dict[int]: List[int]): The actions performed by each agent over the rollout period.
      player_actions (Tuple[int]): The number of actions per agent.
      rollouts (int): The number of rollouts.

    Returns:
      List[List[float]]: The action probabilities for each agent.

    """
    all_probs = {}

    for (ag, action_hist), num_actions in zip(actions_dict.items(), player_actions):
        counts = np.bincount(action_hist, minlength=num_actions)
        probs = counts / rollouts
        all_probs[ag] = probs

    return all_probs


def calc_com_probs(commitments, rollouts):
    """Calculate the empirical commitment probabilities.

    Args:
      commitments (List[int | ndarray]): A list of commitments.
      rollouts (int): The number of rollouts.

    Returns:
      List[float]: The commitment probabilities for each agent.

    """
    com = sum(commitment is not None for commitment in commitments)
    no_com = (rollouts - com)
    return [com / rollouts, no_com / rollouts]


def update(agents, commitment, actions, payoffs):
    """Perform an update for a list of agents.

    Args:
      agents (List[Agent]): A list of agents.
      commitment (int | ndarray): The commitment from the leader.
      actions (List[int]): A list of each action that was chosen, indexed by agent.
      payoffs (List[ndarray]): A list of each payoff that was received, indexed by agent.

    Returns:

    """
    for idx, agent in enumerate(agents):
        agent.update(commitment, actions, payoffs[idx])


def create_agents(experiment, u_tpl, num_agents, player_actions, num_objectives, alpha_q=0.2, alpha_theta=0.005,
                  alpha_q_decay=1, alpha_theta_decay=1, alpha_com=0.005, alpha_com_decay=1):
    """Create a list of commitment agents.

    Args:
        experiment (str): The type of experiment that is run. This is used to determine which agents to create.
        u_tpl (Tuple[callable]): A tuple of utility functions.
        num_agents (int): The number of agents to create.
        player_actions (Tuple[int]): The number of actions per player.
        num_objectives (int): The number of objectives.
        alpha_q (float, optional): The learning rate for Q-values. (Default = 0.2)
        alpha_theta (float, optional): The learning rate for policy parameters. (Default = 0.005)
        alpha_q_decay (float, optional): The decay for the Q-values learning rate. (Default = 1)
        alpha_theta_decay (float, optional): The decay for the policy parameters learning rate. (Default = 1)
        alpha_com (float, optional): The learning rate for a commitment strategy when using optional commitment.
            (Default = 0.005)
        alpha_com_decay (float, optional): The decay for the commitment strategy learning rate when using optional
            commitment. (Default = 1)

    Returns:
        List[Agent]: A list of commitment agents.

    Raises:
        Exception: When the requested agent is unknown.

    """
    agents = []
    for ag, u_str, num_actions in zip(range(num_agents), u_tpl, player_actions):
        u = uf.get_u(u_str)
        if experiment == 'coop_action':
            new_agent = CoopActionAgent(ag, u, num_actions, num_objectives, alpha_q=alpha_q, alpha_theta=alpha_theta,
                                        alpha_q_decay=alpha_q_decay, alpha_theta_decay=alpha_theta_decay)
        elif experiment == 'comp_action':
            new_agent = CompActionAgent(ag, u, num_actions, num_objectives, alpha_q=alpha_q, alpha_theta=alpha_theta,
                                        alpha_q_decay=alpha_q_decay, alpha_theta_decay=alpha_theta_decay)
        elif experiment == 'coop_policy':
            new_agent = CoopPolicyAgent(ag, u, num_actions, num_objectives, alpha_q=alpha_q, alpha_theta=alpha_theta,
                                        alpha_q_decay=alpha_q_decay, alpha_theta_decay=alpha_theta_decay)
        elif experiment == 'best_response':
            new_agent = BestResponseAgent(ag, u, num_actions, num_objectives, alpha_q=alpha_q, alpha_theta=alpha_theta,
                                          alpha_q_decay=alpha_q_decay, alpha_theta_decay=alpha_theta_decay)
        elif experiment == 'non_stationary':
            new_agent = NonStationaryAgent(ag, u, num_actions, num_objectives, alpha_q=alpha_q, alpha_theta=alpha_theta,
                                           alpha_q_decay=alpha_q_decay, alpha_theta_decay=alpha_theta_decay)
        elif experiment == 'opt_coop_action':
            no_com_agent = IndependentActorCriticAgent(u, num_actions, num_objectives, alpha_q=alpha_q,
                                                       alpha_theta=alpha_theta, alpha_q_decay=alpha_q_decay,
                                                       alpha_theta_decay=alpha_theta_decay)
            com_agent = CoopActionAgent(ag, u, num_actions, num_objectives, alpha_q=alpha_q, alpha_theta=alpha_theta,
                                        alpha_q_decay=alpha_q_decay, alpha_theta_decay=alpha_theta_decay)
            new_agent = OptionalComAgent(no_com_agent, com_agent, ag, u, num_actions, num_objectives, alpha_q=alpha_q,
                                         alpha_theta=alpha_com, alpha_q_decay=alpha_q_decay,
                                         alpha_theta_decay=alpha_com_decay)
        elif experiment == 'opt_comp_action':
            no_com_agent = IndependentActorCriticAgent(u, num_actions, num_objectives, alpha_q=alpha_q,
                                                       alpha_theta=alpha_theta, alpha_q_decay=alpha_q_decay,
                                                       alpha_theta_decay=alpha_theta_decay)
            com_agent = CompActionAgent(ag, u, num_actions, num_objectives, alpha_q=alpha_q, alpha_theta=alpha_theta,
                                        alpha_q_decay=alpha_q_decay, alpha_theta_decay=alpha_theta_decay)
            new_agent = OptionalComAgent(no_com_agent, com_agent, ag, u, num_actions, num_objectives, alpha_q=alpha_q,
                                         alpha_theta=alpha_com, alpha_q_decay=alpha_q_decay,
                                         alpha_theta_decay=alpha_com_decay)
        elif experiment == 'opt_coop_policy':
            no_com_agent = IndependentActorCriticAgent(u, num_actions, num_objectives, alpha_q=alpha_q,
                                                       alpha_theta=alpha_theta, alpha_q_decay=alpha_q_decay,
                                                       alpha_theta_decay=alpha_theta_decay)
            com_agent = CoopPolicyAgent(ag, u, num_actions, num_objectives, alpha_q=alpha_q, alpha_theta=alpha_theta,
                                        alpha_q_decay=alpha_q_decay, alpha_theta_decay=alpha_theta_decay)
            new_agent = OptionalComAgent(no_com_agent, com_agent, ag, u, num_actions, num_objectives, alpha_q=alpha_q,
                                         alpha_theta=alpha_com, alpha_q_decay=alpha_q_decay,
                                         alpha_theta_decay=alpha_com_decay)
        else:
            raise Exception(f'No agent of type {experiment} exists')
        agents.append(new_agent)
    return agents


def execute_commitment(payoff_matrices, u_tpl, experiment='coop_action', runs=100, episodes=5000, rollouts=100,
                       alternate=False, alpha_q=0.01, alpha_theta=0.01, alpha_q_decay=1, alpha_theta_decay=1,
                       alpha_com=0.01, alpha_com_decay=1):
    """Execute a commitment experiment.

    Args:
        payoff_matrices (List[ndarray]): A list of payoff matrices representing the MONFG.
        u_tpl (Tuple[callable]): A tuple of utility functions.
        experiment (str, optional): The type of commitment experiment to execute. (Default = 'coop_action')
        runs (int, optional): The number of times to repeat the experiment. (Default = 100)
        episodes (int, optional): The number of episodes in one run of the experiment. (Default = 100)
        rollouts (int, optional): The number of Monte-Carlo simulations at each episode. (Default = 100)
        alternate (bool, optional): Whether to alternate the players between leader and follower. (Default = False)
        alpha_q (float, optional): The learning rate for Q-values. (Default = 0.2)
        alpha_theta (float, optional): The learning rate for policy parameters. (Default = 0.005)
        alpha_q_decay (float, optional): The decay of the learning rate for Q-values. (Default = 1)
        alpha_theta_decay (float, optional): The decay for the learning rate of policy parameters. (Default = 1)
        alpha_com (float, optional): The learning rate for optional commitment. (Default = 0.005)
        alpha_com_decay (float, optional): The decay for the learning rate of commitment. (Default = 1)

    Returns:
        Tuple[Dict, Dict, ndarray, Dict}: A log of payoffs, a log of action probabilities for both agents, a log of the
            state distribution and a log of the commitment probabilities.

    Raises:
        Exception: When the number of players does not equal two.

    """
    num_agents = len(payoff_matrices)

    if num_agents != 2:
        raise Exception(f'Commitment experiments with {num_agents} are currently not supported')

    player_actions = payoff_matrices[0].shape[:-1]
    num_objectives = payoff_matrices[0].shape[-1]

    # Set up logging data structures.
    returns_log = defaultdict(list)
    action_probs_log = defaultdict(list)
    state_dist_log = np.zeros(player_actions)
    com_probs_log = defaultdict(list)

    start = time.time()

    for run in range(runs):
        print("Starting run: ", run)
        agents = create_agents(experiment, u_tpl, num_agents, player_actions, num_objectives, alpha_q=alpha_q,
                               alpha_theta=alpha_theta, alpha_q_decay=alpha_q_decay,
                               alpha_theta_decay=alpha_theta_decay, alpha_com=alpha_com,
                               alpha_com_decay=alpha_com_decay)

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

    return returns_log, action_probs_log, state_dist_log, com_probs_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--game', type=str, default='game5', help="which MONFG game to play")
    parser.add_argument('--u', type=str, default=['u1', 'u2'], nargs='+',
                        help="Which utility functions to use per player")
    parser.add_argument('--experiment', type=str, default='coop_action', help='The experiment to run.')
    parser.add_argument('--alternate', type=bool, default=True, help="Alternate commitment between players.")
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
    alternate = args.alternate
    runs = args.runs
    episodes = args.episodes
    rollouts = args.rollouts
    parent_dir = args.dir

    # Starting the experiments.
    payoff_matrices = games.get_monfg(game)
    data = execute_commitment(payoff_matrices, u, experiment=experiment, runs=runs, episodes=episodes,
                              rollouts=rollouts, alternate=alternate)
    returns_log, action_probs_log, state_dist_log, com_probs_log = data

    # Writing the data to disk.
    num_agents = len(payoff_matrices)
    player_actions = tuple(payoff_matrices[0].shape[:-1])
    path = ex.create_game_path('data', experiment, game, parent_dir=parent_dir)
    save_data(path, experiment, game, num_agents, player_actions, runs, episodes, returns_log=returns_log,
              action_probs_log=action_probs_log, state_dist_log=state_dist_log, com_probs_log=com_probs_log)
