import os

import numpy as np
from jax.nn import softmax

from mo_gt.commitment.best_response_agent import BestResponseAgent
from mo_gt.commitment.comp_action_agent import CompActionAgent
from mo_gt.commitment.coop_action_agent import CoopActionAgent
from mo_gt.commitment.coop_policy_agent import CoopPolicyAgent
from mo_gt.commitment.non_stationary_agent import NonStationaryAgent
from mo_gt.commitment.optional_com_agent import OptionalComAgent
from mo_gt.games.utility_functions import get_u
from mo_gt.learners.indep_actor_critic import IndependentActorCriticAgent
from mo_gt.learners.indep_q import IndependentQAgent
from mo_gt.learners.ja_actor_critic import JointActionActorCriticAgent
from mo_gt.learners.ja_q import JointActionQAgent


def create_game_path(content, experiment, game, parent_dir=None, mkdir=True):
    """Create a new directory based on the given parameters.

    Args:
      content (str): The type of content this directory will hold. Most often this is either 'data' or 'plots'.
      experiment (str): The name of the experiment that is being performed.
      game (str): The game that is experimented on.
      parent_dir (str, optional): Parent directory for data and plots. (Default = None)
      mkdir (bool, optional): Whether to create the directory. (Default = True)

    Returns:
      str: The path that was created.

    """
    if parent_dir is None:
        parent_dir = '.'
    path = f'{parent_dir}/{content}/{experiment}/{game}'
    if mkdir:
        os.makedirs(path, exist_ok=True)

    return path


def make_strat_from_action(action, num_actions):
    """Turn an action into a strategy representation.

    Args:
        action (int): An action.
        num_actions (int): The number of possible actions.

    Returns:
        ndarray: A pure strategy as a numpy array.

    """
    strat = np.zeros(num_actions)
    strat[action] = 1
    return strat


def make_joint_strat(player_id, player_strat, opp_strat):
    """Make joint strategy from the opponent strategy and player strategy.

    Args:
        player_id (int): The id of the player.
        player_strat (ndarray): The strategy of the player.
        opp_strat (List[ndarray]): A list of the strategies of all other players.

    Returns:
        List[ndarray]: A list of strategies with the player's strategy at the correct index.

    """
    opp_strat.insert(player_id, player_strat)
    return opp_strat


def softmax_policy(theta):
    """Take a softmax over an array of parameters.

    Args:
      theta (ndarray): An array of policy parameters.

    Returns:
      ndarray: A probability distribution over actions as a policy.

    """
    policy = np.asarray(softmax(theta), dtype=float)
    policy = policy / np.sum(policy)
    return policy


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


def create_agents(experiment, u_tpl, num_agents, player_actions, num_objectives, alpha_q=0.01, alpha_theta=0.01,
                  alpha_fq=0.01, alpha_ftheta=0.01, alpha_cq=0.01, alpha_ctheta=0.01, alpha_q_decay=1,
                  alpha_theta_decay=1, alpha_com_decay=1, epsilon=1, epsilon_decay=0.995, min_epsilon=0.1, rng=None):
    """Create a list of agents.

    Args:
        experiment (str): The type of experiment that is run. This is used to determine which agents to create.
        u_tpl (Tuple[str]): A tuple of utility functions.
        num_agents (int): The number of agents to create.
        player_actions (Tuple[int]): The number of actions per player.
        num_objectives (int): The number of objectives.
        alpha_q (float, optional): The learning rate for Q-values. (Default = 0.01)
        alpha_theta (float, optional): The learning rate for policy parameters. (Default = 0.01)
        alpha_fq (float, optional): The learning rate for follower Q-values. (Default = 0.01)
        alpha_ftheta (float, optional): The learning rate for follower policy parameters. (Default = 0.01)
        alpha_cq (float, optional): The learning rate for optional commitment Q-values. (Default = 0.01)
        alpha_ctheta (float, optional): The learning rate for optional commitment policy parameters. (Default = 0.01)
        alpha_q_decay (float, optional): The decay for the Q-values learning rate. (Default = 1)
        alpha_theta_decay (float, optional): The decay for the policy parameters learning rate. (Default = 1)
        alpha_com_decay (float, optional): The decay for the optional commitment strategy learning rate. (Default = 1)
        epsilon (float, optional): The exploration rate for a Q-learner agent. (Default = 1)
        epsilon_decay (float, optional): The decay for the exploration rate. (Default = 0.995)
        min_epsilon (float, optional): The minimum value for the exploration rate. (Default = 0.1)
        rng (Generator, optional): A random number generator. (Default = None)

    Returns:
        List[Agent]: A list of agents.

    Raises:
        Exception: When the requested agent is unknown in the context of the experiment.

    """
    agents = []
    for ag, u_str, num_actions in zip(range(num_agents), u_tpl, player_actions):
        u = get_u(u_str)
        if experiment == 'indep_ac':
            new_agent = IndependentActorCriticAgent(u, num_actions, num_objectives, alpha_q=alpha_q,
                                                    alpha_theta=alpha_theta, alpha_q_decay=alpha_q_decay,
                                                    alpha_theta_decay=alpha_theta_decay)
        elif experiment == 'indep_q':
            new_agent = IndependentQAgent(u, num_actions, num_objectives, alpha_q=alpha_q, alpha_q_decay=alpha_q_decay,
                                          epsilon=epsilon, epsilon_decay=epsilon_decay, min_epsilon=min_epsilon)
        elif experiment == 'ja_ac':
            new_agent = JointActionActorCriticAgent(ag, u, num_actions, num_objectives, player_actions, alpha_q=alpha_q,
                                                    alpha_theta=alpha_theta, alpha_q_decay=alpha_q_decay,
                                                    alpha_theta_decay=alpha_theta_decay)
        elif experiment == 'ja_q':
            new_agent = JointActionQAgent(ag, u, num_actions, num_objectives, player_actions, alpha_q=alpha_q,
                                          alpha_q_decay=alpha_q_decay, epsilon=epsilon, epsilon_decay=epsilon_decay,
                                          min_epsilon=min_epsilon)
        elif experiment == 'coop_action':
            new_agent = CoopActionAgent(ag, u, num_actions, num_objectives, alpha_q=alpha_q, alpha_theta=alpha_theta,
                                        alpha_q_decay=alpha_q_decay, alpha_theta_decay=alpha_theta_decay, rng=rng)
        elif experiment == 'comp_action':
            new_agent = CompActionAgent(ag, u, num_actions, num_objectives, alpha_lq=alpha_q,
                                        alpha_ltheta=alpha_theta, alpha_fq=alpha_fq, alpha_ftheta=alpha_ftheta,
                                        alpha_q_decay=alpha_q_decay, alpha_theta_decay=alpha_theta_decay, rng=rng)
        elif experiment == 'coop_policy':
            new_agent = CoopPolicyAgent(ag, u, num_actions, num_objectives, alpha_q=alpha_q, alpha_theta=alpha_theta,
                                        alpha_q_decay=alpha_q_decay, alpha_theta_decay=alpha_theta_decay, rng=rng)
        elif experiment == 'best_response':
            new_agent = BestResponseAgent(ag, u, num_actions, num_objectives, alpha_q=alpha_q,
                                          alpha_theta=alpha_theta, alpha_q_decay=alpha_q_decay,
                                          alpha_theta_decay=alpha_theta_decay, rng=rng)
            new_agent.set_leader_utility(get_u(u_tpl[0]))
        elif experiment == 'non_stationary':
            new_agent = NonStationaryAgent(ag, u, num_actions, num_objectives, alpha_q=alpha_q,
                                           alpha_theta=alpha_theta, alpha_q_decay=alpha_q_decay,
                                           alpha_theta_decay=alpha_theta_decay, rng=rng)
            new_agent.set_opponent_actions(player_actions[abs(1 - ag)])
        elif experiment == 'opt_coop_action':
            no_com_agent = IndependentActorCriticAgent(u, num_actions, num_objectives, alpha_q=alpha_q,
                                                       alpha_theta=alpha_theta, alpha_q_decay=alpha_q_decay,
                                                       alpha_theta_decay=alpha_theta_decay, rng=rng)
            com_agent = CoopActionAgent(ag, u, num_actions, num_objectives, alpha_q=alpha_q, alpha_theta=alpha_theta,
                                        alpha_q_decay=alpha_q_decay, alpha_theta_decay=alpha_theta_decay, rng=rng)
            new_agent = OptionalComAgent(no_com_agent, com_agent, ag, u, num_actions, num_objectives, alpha_q=alpha_cq,
                                         alpha_theta=alpha_ctheta, alpha_q_decay=alpha_q_decay,
                                         alpha_theta_decay=alpha_com_decay, rng=rng)
        elif experiment == 'opt_comp_action':
            no_com_agent = IndependentActorCriticAgent(u, num_actions, num_objectives, alpha_q=alpha_q,
                                                       alpha_theta=alpha_theta, alpha_q_decay=alpha_q_decay,
                                                       alpha_theta_decay=alpha_theta_decay, rng=rng)
            com_agent = CompActionAgent(ag, u, num_actions, num_objectives, alpha_lq=alpha_q,
                                        alpha_ltheta=alpha_theta, alpha_fq=alpha_fq, alpha_ftheta=alpha_ftheta,
                                        alpha_q_decay=alpha_q_decay, alpha_theta_decay=alpha_theta_decay, rng=rng)
            new_agent = OptionalComAgent(no_com_agent, com_agent, ag, u, num_actions, num_objectives, alpha_q=alpha_cq,
                                         alpha_theta=alpha_ctheta, alpha_q_decay=alpha_q_decay,
                                         alpha_theta_decay=alpha_com_decay, rng=rng)
        elif experiment == 'opt_coop_policy':
            no_com_agent = IndependentActorCriticAgent(u, num_actions, num_objectives, alpha_q=alpha_q,
                                                       alpha_theta=alpha_theta, alpha_q_decay=alpha_q_decay,
                                                       alpha_theta_decay=alpha_theta_decay, rng=rng)
            com_agent = CoopPolicyAgent(ag, u, num_actions, num_objectives, alpha_q=alpha_q, alpha_theta=alpha_theta,
                                        alpha_q_decay=alpha_q_decay, alpha_theta_decay=alpha_theta_decay, rng=rng)
            new_agent = OptionalComAgent(no_com_agent, com_agent, ag, u, num_actions, num_objectives, alpha_q=alpha_cq,
                                         alpha_theta=alpha_ctheta, alpha_q_decay=alpha_q_decay,
                                         alpha_theta_decay=alpha_com_decay, rng=rng)
        else:
            raise Exception(f'No agent of type {experiment} exists')
        agents.append(new_agent)
    return agents
