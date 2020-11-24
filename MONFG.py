import time
import argparse
import pandas as pd
from utils import *
from games import *
from QLearnerESR import QLearnerESR
from QLearnerSER import QLearnerSER


def select_actions(agents):
    """
    This function selects an action from each agent's policy.
    :param agents: The list of agents.
    :return: A list of selected actions.
    """
    selected = []
    for agent in agents:
        selected.append(agent.select_action())
    return selected


def calc_payoffs(agents, actions, payoff_matrix):
    """
    This function will calculate the payoffs of the agents.
    :param agents: The list of agents.
    :param actions: The action that each agent chose.
    :param payoff_matrix: The payoff matrix.
    :return: A list of received payoffs.
    """
    payoffs = []
    for agent in agents:
        payoffs.append(payoff_matrix[actions[0]][actions[1]])  # Append the payoffs from the actions.
    return payoffs


def calc_returns(action_probs, criterion, payoff_matrix):
    """
    This function will calculate the expected returns under the given criterion.
    :param action_probs: The current action probabilities of the agents.
    :param criterion: The multi-objective criterion. Either SER or ESR.
    :param payoff_matrix: The payoff matrix.
    :return: A list of expected returns.
    """
    policy1 = action_probs[0]
    policy2 = action_probs[1]

    if criterion == 'SER':
        expected_returns = policy2 @ (policy1 @ payoff_matrix)  # Calculate the expected returns.
        ser1 = u1(expected_returns)  # Scalarise the expected returns.
        ser2 = u2(expected_returns)

        return [ser1, ser2]
    else:
        scalarised_returns1 = scalarise_matrix(payoff_matrix, u1)  # Scalarise the possible returns.
        scalarised_returns2 = scalarise_matrix(payoff_matrix, u2)
        esr1 = policy2 @ (policy1 @ scalarised_returns1)  # Take the expected value over them.
        esr2 = policy2 @ (policy1 @ scalarised_returns2)

        return [esr1, esr2]


def get_action_probs(agents):
    """
    This function gets the current action probabilities from each agent.
    :param agents: A list of agents.
    :return: A list of their action probabilities.
    """
    action_probs = []
    for agent in agents:
        action_probs.append(agent.strategy)
    return action_probs


def decay_params(agents, alpha_decay, epsilon_decay):
    """
    This function decays the parameters of the Q-learning algorithm used in each agent.
    :param agents: A list of agents.
    :param alpha_decay: The factor by which to decay alpha.
    :param epsilon_decay: The factor by which to decay epsilon.
    :return: /
    """
    for agent in agents:
        agent.alpha *= alpha_decay
        agent.epsilon *= epsilon_decay


def update(agents, actions, payoffs):
    """
    This function gets called after every episode to update the policy of every agent.
    :param agents: A list of agents.
    :param actions: A list of each action that was chosen, indexed by agent.
    :param payoffs: A list of each payoff that was received, indexed by agent.
    :return:
    """
    for idx, agent in enumerate(agents):
        agent.update(actions[idx], payoffs[idx])


def reset(num_agents, num_actions, num_objectives, alpha, epsilon, opt=False, rand_prob=False):
    """
    Ths function will create fresh agents that can be used in a new trial.
    :param num_agents: The number of agents to create.
    :param num_actions: The number of actions each agent can take.
    :param num_objectives: The number of objectives they have.
    :param alpha: The learning rate.
    :param epsilon: The epsilon used in their epsilon-greedy strategy.
    :param opt: A boolean that decides on optimistic initialization of the Q-tables.
    :param rand_prob: A boolean that decides on random initialization for the mixed strategy.
    :return:
    """
    agents = []
    for ag in range(num_agents):
        u, du = get_u_and_du(ag + 1)  # The utility function and derivative of the utility function for this agent.
        if criterion == 'SER':
            new_agent = QLearnerSER(u, alpha, epsilon, num_actions, num_objectives, opt, rand_prob)
        else:
            new_agent = QLearnerESR(u, alpha, epsilon, num_actions, num_objectives, opt, rand_prob)
        agents.append(new_agent)
    return agents


def run_experiment(runs, episodes, criterion, payoff_matrix, opt_init, rand_prob):
    """
    This function will run the requested experiment.
    :param runs: The number of different runs.
    :param episodes: The number of episodes in each run.
    :param criterion: The multi-objective optimisation criterion to use.
    :param payoff_matrix: The payoff matrix for the game.
    :param opt_init: A boolean that decides on optimistic initialization of the Q-tables.
    :param rand_prob: A boolean that decides on random initialization for the mixed strategy.
    :return: A log of payoffs, a log for action probabilities for both agents and a log of the state distribution.
    """
    # Setting hyperparameters.
    num_agents = 2
    num_actions = payoff_matrix.shape[0]
    num_objectives = 2
    epsilon = 0.1
    epsilon_decay = 0.999
    alpha = 0.05
    alpha_decay = 1

    # Setting up lists containing the results.
    payoffs_log1 = []
    payoffs_log2 = []
    act_hist_log = [[], []]
    state_dist_log = np.zeros((num_actions, num_actions))

    start = time.time()

    for run in range(runs):
        print("Starting run: ", run)
        agents = reset(num_agents, num_actions, num_objectives, alpha, epsilon, opt_init, rand_prob)

        for episode in range(episodes):
            # Run one episode.
            actions = select_actions(agents)
            payoffs = calc_payoffs(agents, actions, payoff_matrix)
            update(agents, actions, payoffs)  # Update the current strategy based on the returns.
            decay_params(agents, alpha_decay, epsilon_decay)  # Decay the parameters after the episode is finished.

            # Get the necessary results from this episode.
            probs = get_action_probs(agents)  # Get the current action probabilities of the agents.
            returns = calc_returns(probs, criterion, payoff_matrix)  # Calculate the SER/ESR of the current strategies.

            # Append the returns under the criterion and the action probabilities to the logs.
            returns1, returns2 = returns
            probs1, probs2 = probs
            payoffs_log1.append([episode, run, returns1])
            payoffs_log2.append([episode, run, returns2])

            if num_actions == 2:
                act_hist_log[0].append([episode, run, probs1[0], probs1[1], 0])
                act_hist_log[1].append([episode, run, probs2[0], probs2[1], 0])
            elif num_actions == 3:
                act_hist_log[0].append([episode, run, probs1[0], probs1[1], probs1[2]])
                act_hist_log[1].append([episode, run, probs2[0], probs2[1], probs2[2]])
            else:
                Exception("This number of actions is not yet supported")

            # If we are in the last 10% of episodes we build up a state distribution log.
            if episode >= 0.9 * episodes:
                state_dist_log[actions[0], actions[1]] += 1

    end = time.time()
    elapsed_mins = (end - start) / 60.0
    print("Minutes elapsed: " + str(elapsed_mins))

    return payoffs_log1, payoffs_log2, act_hist_log, state_dist_log


def create_data_dir(criterion, game, opt_init, rand_prob):
    """
    This function will create a new directory based on the given parameters.
    :param criterion: The multi-objective optimisation criterion.
    :param game: The current game that is being played.
    :param opt_init: A boolean that decides on optimistic initialization of the Q-tables.
    :param rand_prob: A boolean that decides on random initialization for the mixed strategy.
    :return: The path that was created.
    """
    path = f'data/{criterion}/{game}'

    if opt_init:
        path += '/opt_init'
    else:
        path += '/zero_init'

    if rand_prob:
        path += '/opt_rand'
    else:
        path += '/opt_eq'

    print("Creating data path: " + repr(path))
    mkdir_p(path)

    return path


def save_data(path, name, payoffs_log1, payoffs_log2, act_hist_log, state_dist_log, runs, episodes):
    """
    This function will save all of the results to disk in CSV format for later analysis.
    :param path: The path to the directory in which all files will be saved.
    :param name: The name of the experiment.
    :param payoffs_log1: The payoff logs for agent 1.
    :param payoffs_log2: The payoff logs for agent 2.
    :param act_hist_log: The action logs for both agents.
    :param state_dist_log: The state distribution log in the last 10% of episodes.
    :param runs: The number of trials that were ran.
    :param episodes: The number of episodes in each run.
    :return: /
    """
    print("Saving data to disk")
    columns = ['Episode', 'Trial', 'Payoff']
    df1 = pd.DataFrame(payoffs_log1, columns=columns)
    df2 = pd.DataFrame(payoffs_log2, columns=columns)

    df1.to_csv(f'{path}/agent1_{name}.csv', index=False)
    df2.to_csv(f'{path}/agent2_{name}.csv', index=False)

    columns = ['Episode', 'Trial', 'Action 1', 'Action 2', 'Action 3']
    df1 = pd.DataFrame(act_hist_log[0], columns=columns)
    df2 = pd.DataFrame(act_hist_log[1], columns=columns)

    df1.to_csv(f'{path}/agent1_probs_{name}.csv', index=False)
    df2.to_csv(f'{path}/agent2_probs_{name}.csv', index=False)

    state_dist_log /= runs * (0.1 * episodes)
    df = pd.DataFrame(state_dist_log)
    df.to_csv(f'{path}/states_{name}.csv', index=False, header=False)
    print("Finished saving data to disk")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-game', type=str, default='game1', choices=['game1', 'game2', 'game3', 'game4', 'game5'],
                        help="which MONFG game to play")
    parser.add_argument('-criterion', type=str, default='SER', choices=['SER', 'ESR'],
                        help="optimization criterion to use")

    parser.add_argument('-name', type=str, default='no_comms', help='The name under which to save the results')
    parser.add_argument('-runs', type=int, default=100, help="number of trials")
    parser.add_argument('-episodes', type=int, default=5000, help="number of episodes")

    # Optimistic initialization can encourage exploration.
    parser.add_argument('-opt_init', action='store_true', help="optimistic initialization")
    parser.add_argument('-rand_prob', action='store_true', help="rand init for optimization prob")

    args = parser.parse_args()

    # Extracting the arguments.
    game = args.game
    criterion = args.criterion
    name = args.name
    runs = args.runs
    episodes = args.episodes
    opt_init = args.opt_init
    rand_prob = args.rand_prob

    # Starting the experiments.
    payoff_matrix = get_payoff_matrix(game)
    data = run_experiment(runs, episodes, criterion, payoff_matrix, opt_init, rand_prob)
    payoffs_log1, payoffs_log2, act_hist_log, state_dist_log = data

    # Writing the data to disk.
    path = create_data_dir(criterion, game, opt_init, rand_prob)
    save_data(path, name, payoffs_log1, payoffs_log2, act_hist_log, state_dist_log, runs, episodes)
