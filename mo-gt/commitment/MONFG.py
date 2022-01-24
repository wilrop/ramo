import time
import argparse

import numpy as np
import pandas as pd

from utils_learn import *
from games import *
from no_com_agent import NoComAgent
from comp_action_agent import CompActionAgent
from coop_action_agent import CoopActionAgent
from coop_policy_agent import CoopPolicyAgent
from optional_com_agent import OptionalComAgent
from best_response_agent import BestResponseAgent
from non_stationary_agent import NonStationaryAgent


def get_communicator(episode, agents, alternate=False):
    """
    This function selects the communicator.
    :param episode: The current episode.
    :param agents: The agents in the game.
    :param alternate: Alternate the leader or always the same.
    :return: The id of the communicating agent and the communicating agent itself.
    """
    if alternate:
        communicator = episode % len(agents)
    else:
        communicator = 0
    communicating_agent = agents[communicator]
    return communicator, communicating_agent


def select_actions(agents, message):
    """
    This function selects an action from each agent's policy.
    :param agents: The list of agents.
    :param message: The message from the leader.
    :return: A list of selected actions.
    """
    selected = []
    for agent in agents:
        selected.append(agent.select_action(message))
    return selected


def calc_payoffs(agents, actions, payoff_matrices):
    """
    This function will calculate the payoffs of the agents.
    :param agents: The list of agents.
    :param actions: The action that each agent chose.
    :param payoff_matrices: The payoff matrices.
    :return: A list of received payoffs.
    """
    payoffs = []
    for payoff_matrix in payoff_matrices:
        payoffs.append(payoff_matrix[actions[0]][actions[1]])  # Append the payoffs from the actions.
    return payoffs


def calc_returns(payoffs, agents, rollouts):
    """
    This function will calculate the scalarised expected returns for each agent.
    :param payoffs: The payoffs obtained by the agents.
    :param agents: The agents in this experiment
    :param rollouts: The amount of rollouts that were performed.
    :return: A list of scalarised expected returns.
    """
    returns = []
    for idx, payoff_hist in enumerate(payoffs):
        payoff_sum = np.sum(payoff_hist, axis=0)
        avg_payoff = payoff_sum / rollouts
        ser = agents[idx].u(avg_payoff)
        returns.append(ser)
    return returns


def calc_action_probs(actions, num_actions, rollouts):
    """
    This function will calculate the empirical action probabilities.
    :param actions: The actions performed by each agent over the rollout period.
    :param num_actions: The number of possible actions.
    :param rollouts: The number of rollouts that were performed.
    :return: The action probabilities for each agent.
    """
    all_probs = []

    for action_hist in actions:
        probs = np.zeros(num_actions)

        for action in action_hist:
            probs[action] += 1

        probs = probs / rollouts
        all_probs.append(probs)

    return all_probs


def calc_com_probs(messages, rollouts):
    """
    This function will calculate the empirical communication probabilities.
    :param messages: The messages that were sent.
    :param rollouts: The number of rollouts that were performed.
    :return: The communication probabilities for each agent.
    """
    com = sum(message is not None for message in messages)
    no_com = (rollouts - com)
    return [com / rollouts, no_com / rollouts]


def update(agents, communicator, message, actions, payoffs):
    """
    This function gets called after every episode so that agents can update their internal mechanisms.
    :param agents: A list of agents.
    :param communicator: The id of the communicating agent.
    :param message: The message that was sent.
    :param actions: A list of each action that was chosen, indexed by agent.
    :param payoffs: A list of each payoff that was received, indexed by agent.
    :return:
    """
    for idx, agent in enumerate(agents):
        agent.update(communicator, message, actions, payoffs[idx])


def reset(experiment, num_agents, u_lst, num_actions, num_objectives, alpha_q, alpha_theta, alpha_msg, alpha_decay,
          opt=False):
    """
    This function will create new agents that can be used in a new trial.
    :param experiment: The type of experiments we are running.
    :param num_agents: The number of agents to create.
    :param u_lst: A list of utility functions to use per agent.
    :param num_actions: The number of actions each agent can take.
    :param num_objectives: The number of objectives they have.
    :param alpha_q: The learning rate for the Q values.
    :param alpha_theta: The learning rate for theta.
    :param alpha_msg: The learning rate for learning a messaging strategy in the optional communication experiments.
    :param alpha_decay: The learning rate decay.
    :param opt: A boolean that decides on optimistic initialization of the Q-tables.
    :return:
    """
    agents = []
    for ag, u_str in zip(range(num_agents), u_lst):
        u, du = get_u_and_du(u_str)  # The utility function and derivative of the utility function for this agent.
        if experiment == 'no_com':
            new_agent = NoComAgent(ag, u, du, alpha_q, alpha_theta, alpha_decay, num_actions, num_objectives, opt)
        elif experiment == 'comp_action':
            new_agent = CompActionAgent(ag, u, du, alpha_q, alpha_theta, alpha_decay, num_actions, num_objectives, opt)
        elif experiment == 'best_response':
            new_agent = BestResponseAgent(ag, u, du, alpha_q, alpha_theta, alpha_decay, num_actions, num_objectives,
                                          opt)
        elif experiment == 'non_stationary':
            new_agent = NonStationaryAgent(ag, u, du, alpha_q, alpha_theta, alpha_decay, num_actions, num_objectives,
                                           opt)
        elif experiment == 'coop_action':
            new_agent = CoopActionAgent(ag, u, du, alpha_q, alpha_theta, alpha_decay, num_actions, num_objectives, opt)
        elif experiment == 'coop_policy':
            new_agent = CoopPolicyAgent(ag, u, du, alpha_q, alpha_theta, alpha_decay, num_actions, num_objectives, opt)
        elif experiment == 'opt_comp_action':
            no_com_agent = NoComAgent(ag, u, du, alpha_q, alpha_theta, alpha_decay, num_actions, num_objectives, opt)
            com_agent = CompActionAgent(ag, u, du, alpha_q, alpha_theta, alpha_decay, num_actions, num_objectives, opt)
            new_agent = OptionalComAgent(no_com_agent, com_agent, ag, u, du, alpha_q, alpha_msg, alpha_decay,
                                         num_objectives, opt)
        elif experiment == 'opt_coop_action':
            no_com_agent = NoComAgent(ag, u, du, alpha_q, alpha_theta, alpha_decay, num_actions, num_objectives, opt)
            com_agent = CoopActionAgent(ag, u, du, alpha_q, alpha_theta, alpha_decay, num_actions, num_objectives, opt)
            new_agent = OptionalComAgent(no_com_agent, com_agent, ag, u, du, alpha_q, alpha_msg, alpha_decay,
                                         num_objectives, opt)
        elif experiment == 'opt_coop_policy':
            no_com_agent = NoComAgent(ag, u, du, alpha_q, alpha_theta, alpha_decay, num_actions, num_objectives, opt)
            com_agent = CoopPolicyAgent(ag, u, du, alpha_q, alpha_theta, alpha_decay, num_actions, num_objectives, opt)
            new_agent = OptionalComAgent(no_com_agent, com_agent, ag, u, du, alpha_q, alpha_msg, alpha_decay,
                                         num_objectives, opt)
        else:
            raise Exception(f'No experiment of type {experiment} exists')
        agents.append(new_agent)
    return agents


def run_experiment(experiment, runs, episodes, rollouts, payoff_matrices, u, alternate, opt_init):
    """
    This function will run the requested experiment.
    :param experiment: The type of experiment we are running.
    :param runs: The number of different runs.
    :param episodes: The number of episodes in each run.
    :param rollouts: The rollout period for the policies.
    :param payoff_matrices: The payoff matrices for the game.
    :param u: A list of utility functions to use for the agents.
    :param alternate: Alternate commitment between players.
    :param opt_init: A boolean that decides on optimistic initialization of the Q-tables.
    :return: A log of payoffs, a log for action probabilities for both agents and a log of the state distribution.
    """
    # Setting hyperparameters.
    num_agents = 2
    num_actions = payoff_matrices[0].shape[0]
    num_objectives = 2
    alpha_q = 0.2
    alpha_theta = 0.005
    alpha_msg = 0.005
    alpha_decay = 1

    # Setting up lists containing the results.
    returns_log = [[] for _ in range(num_agents)]
    action_probs_log = [[] for _ in range(num_agents)]
    com_probs_log = [[] for _ in range(num_agents)]
    state_dist_log = np.zeros((num_actions, num_actions))

    start = time.time()

    for run in range(runs):
        print("Starting run: ", run)
        agents = reset(experiment, num_agents, u, num_actions, num_objectives, alpha_q, alpha_theta, alpha_msg,
                       alpha_decay, opt_init)

        for episode in range(episodes):
            # We keep the actions and payoffs of this episode so that we can later calculate the SER.
            ep_actions = [[] for _ in range(num_agents)]
            ep_payoffs = [[] for _ in range(num_agents)]
            ep_messages = []

            communicator, communicating_agent = get_communicator(episode, agents, alternate)

            for rollout in range(rollouts):  # Required to evaluate the SER and action probabilities.
                message = communicating_agent.get_message()
                actions = select_actions(agents, message)
                payoffs = calc_payoffs(agents, actions, payoff_matrices)

                # Log the results of this roll
                for idx in range(num_agents):
                    ep_actions[idx].append(actions[idx])
                    ep_payoffs[idx].append(payoffs[idx])
                ep_messages.append(message)

            # Update the agent after the episode
            # We use the last action and payoff to update the agent. It doesn't really matter which rollout we select
            # to update our agent as the agent doesn't learn any new information during the rollout.
            last_actions = np.array(ep_actions)[:, -1]
            last_payoffs = np.array(ep_payoffs)[:, -1]
            last_message = ep_messages[-1]
            update(agents, communicator, last_message, last_actions, last_payoffs)  # Update the agents.

            # Get the necessary results from this episode.
            action_probs = calc_action_probs(ep_actions, num_actions, rollouts)
            returns = calc_returns(ep_payoffs, agents, rollouts)
            com_probs = calc_com_probs(ep_messages, rollouts)

            # Append the logs.
            for idx in range(num_agents):
                returns_log[idx].append([run, episode, returns[idx]])
                prob_log = [run, episode] + action_probs[idx].tolist()
                action_probs_log[idx].append(prob_log)
            com_log = [run, episode] + com_probs
            com_probs_log[communicator].append(com_log)

            # If we are in the last 10% of episodes we build up a state distribution log.
            # This code is specific to two player games.
            if episode >= 0.9 * episodes:
                state_dist = np.zeros((num_actions, num_actions))
                for a1, a2 in zip(ep_actions[0], ep_actions[1]):
                    state_dist[a1, a2] += 1
                state_dist /= rollouts
                state_dist_log += state_dist

    end = time.time()
    elapsed_mins = (end - start) / 60.0
    print("Minutes elapsed: " + str(elapsed_mins))

    return returns_log, action_probs_log, com_probs_log, state_dist_log


def save_data(path, name, returns_log, action_probs_log, com_probs_log, state_dist_log, runs, episodes):
    """
    This function will save all of the results to disk in CSV format for later analysis.
    :param path: The path to the directory in which all files will be saved.
    :param name: The name of the experiment.
    :param returns_log: The log for the returns.
    :param action_probs_log: The log for the action probabilities.
    :param action_probs_log: The log for the communication probabilities.
    :param state_dist_log: The state distribution log in the last 10% of episodes.
    :param runs: The number of trials that were ran.
    :param episodes: The number of episodes in each run.
    :return: /
    """
    print("Saving data to disk")
    num_agents = len(returns_log)  # Extract the number of agents that were in the experiment.
    num_actions = len(action_probs_log[0][0]) - 2  # Extract the number of actions that were possible in the experiment.
    returns_columns = ['Trial', 'Episode', 'Payoff']
    action_columns = [f'Action {a + 1}' for a in range(num_actions)]
    action_columns = ['Trial', 'Episode'] + action_columns
    com_columns = ['Trial', 'Episode', 'Communication', 'No communication']

    for idx in range(num_agents):
        df_r = pd.DataFrame(returns_log[idx], columns=returns_columns)
        df_a = pd.DataFrame(action_probs_log[idx], columns=action_columns)
        df_r.to_csv(f'{path}/{name}_{game}_A{idx + 1}_returns.csv', index=False)
        df_a.to_csv(f'{path}/{name}_{game}_A{idx + 1}_probs.csv', index=False)

    if name in ['opt_comp_action', 'opt_coop_action', 'opt_coop_policy']:
        for idx in range(num_agents):
            df = pd.DataFrame(com_probs_log[idx], columns=com_columns)
            df.to_csv(f'{path}/{name}_{game}_A{idx + 1}_com.csv', index=False)

    state_dist_log /= runs * (0.1 * episodes)
    df = pd.DataFrame(state_dist_log)
    df.to_csv(f'{path}/{name}_{game}_states.csv', index=False, header=False)
    print("Finished saving data to disk")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--game', type=str, default='game9', help="which MONFG game to play")
    parser.add_argument('--u', type=str, default=['u1', 'u2'], nargs='+',
                        help="Which utility functions to use per player")
    parser.add_argument('--experiment', type=str, default='comp_action', help='The experiment to run.')
    parser.add_argument('--alternate', type=bool, default=False, help="Alternate commitment between players.")
    parser.add_argument('--runs', type=int, default=100, help="number of trials")
    parser.add_argument('--episodes', type=int, default=5000, help="number of episodes")
    parser.add_argument('--rollouts', type=int, default=100, help="Rollout period for the policies")

    # Optimistic initialization can encourage exploration.
    parser.add_argument('--opt_init', action='store_true', help="optimistic initialization")

    args = parser.parse_args()

    # Extracting the arguments.
    game = args.game
    u = args.u
    experiment = args.experiment
    alternate = args.alternate
    runs = args.runs
    episodes = args.episodes
    rollouts = args.rollouts
    opt_init = args.opt_init

    # Starting the experiments.
    payoff_matrices = get_monfg(game)
    data = run_experiment(experiment, runs, episodes, rollouts, payoff_matrices, u, alternate, opt_init)
    returns_log, action_probs_log, com_probs_log, state_dist_log = data

    # Writing the data to disk.
    path = create_game_path('data', experiment, game, opt_init)
    mkdir_p(path)
    save_data(path, experiment, returns_log, action_probs_log, com_probs_log, state_dist_log, runs, episodes)
