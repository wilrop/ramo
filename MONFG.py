import time
import argparse
import pandas as pd
from utils import *
from games import *
from QLearnerESR import QLearnerESR, calc_utility
from QLearnerSER import QLearnerSER
from collections import Counter


def select_actions(agents):
    """
    This function will select actions from the starting message for each agent
    :return: The actions that were selected.
    """
    selected = []
    for agent in agents:
        selected.append(agent.select_action_mixed_nonlinear(message))
    return selected


def calc_payoffs(agents, selected_actions, payoff_matrix):
    """
    This function will calculate the payoffs of the agents.
    :return: /
    """
    payoffs = []
    for agent in agents:
        payoffs.append(payoff_matrix[selected_actions[0]][selected_actions[1]])  # Append the payoffs from the actions.
    return payoffs


def calc_returns(payoffs, criterion):
    if criterion == 'SER':
        payoffs1 = payoffs[0]
        payoffs2 = payoffs[1]
        returns1 = u1(np.mean(payoffs1, axis=0))
        returns2 = u2(np.mean(payoffs2, axis=0))

        return [returns1, returns2]
    else:
        payoffs1 = payoffs[0]
        payoffs2 = payoffs[1]
        returns1 = np.mean([u1(payoff) for payoff in payoffs1])
        returns2 = np.mean([u2(payoff) for payoff in payoffs2])

        return [returns1, returns2]


def decay_params(agents, alpha_decay, epsilon_decay):
    """
    Decay the parameters of the Q-learning algorithm.
    :return: /
    """
    for agent in agents:
        agent.alpha *= alpha_decay
        agent.epsilon *= epsilon_decay


def update(agents, selected_actions, returns):
    """
    This function gets called after every episode to update the Q-tables.
    :return: /
    """
    for idx, agent in enumerate(agents):
        agent.update_q_table(message, selected_actions[idx], returns[idx])
        agent.update_joint_table(selected_actions, returns[idx])


def get_message(agents, communicate, episode):
    """
    This function prepares the communication for this episode. This is the preferred action of a specific agent.
    :param ep: The current episode.
    :return: The preferred joint action.
    """
    communicator = episode % len(agents)
    if communicate:
        message = agents[communicator].pref_joint_action()
    else:
        message = 0
    return communicator, message


def do_episode(agents, communicate, episode):
    """
    Runs an entire episode of the game.
    :param episode: The current episode.
    :return: The actions that were selected.
    """
    global selected_actions, payoffs, current_states
    if communicate:
        get_message(episode)
    selected_actions = select_actions(agents)
    payoffs = calc_payoffs(payoff_matrix, selected_actions)
    update()
    decay_params()
    return selected_actions, payoffs


def do_rollout(agents, communicate, episode):
    if communicate:
        get_message(episode)
    selected_actions = select_actions(agents)
    payoffs = calc_payoffs(payoff_matrix, selected_actions)
    return selected_actions, payoffs


def reset(num_agents, num_actions, num_objectives, alpha, epsilon, opt=False, rand_prob=False):
    """
    This function will reset all variables for the new episode.
    :param opt: Boolean that decides on optimistic initialization of the Q-tables.
    :param rand_prob: Boolean that decides on random initialization for the mixed  strategy.
    :return: A list of agents.
    """
    agents = []
    for ag in range(num_agents):
        if criterion == 'SER':
            new_agent = QLearnerSER(ag, alpha, 0, epsilon, num_actions, num_actions, num_objectives, opt, rand_prob)
        else:
            new_agent = QLearnerESR(ag, alpha, 0, epsilon, num_actions, num_actions, num_objectives, opt, rand_prob)
        agents.append(new_agent)
    return agents


def update_state_dist(selected_actions, state_dist_log):
    actions1 = selected_actions[0]
    actions2 = selected_actions[1]
    for action1, action2 in zip(actions1, actions2):
        state_dist_log[action1, action2] += 1
    return state_dist_log


def run_experiment(runs, episodes, rollout, criterion, communicate, payoff_matrix, opt_init, rand_prob):
    # Setting hyperparameters.
    num_agents = 2
    num_actions = payoff_matrix.shape[0]
    num_objectives = 2
    epsilon = 0.9
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
            episode_payoffs = [[], []]
            action_hist1 = np.array([0, 0, 0])
            action_hist2 = np.array([0, 0, 0])

            communicator, message = get_message(agents, communicate, episode)

            for r in range(rollout):
                actions, payoffs = do_rollout(agents, communicate, episode)
                episode_payoffs[0].append(payoffs[0])
                episode_payoffs[1].append(payoffs[1])
                action_hist1[actions[0]] += 1
                action_hist2[actions[1]] += 1

            # Transform the action history for this episode to probabilities.
            action_hist1 /= rollout
            action_hist2 /= rollout

            returns = calc_returns(episode_payoffs, criterion)  # Calculate the SER or ESR of the current strategy
            update(agents, actions, returns)  # Update the current strategy based on the returns.
            decay_params(agents, alpha_decay, epsilon_decay)  # Decay the parameters after the rollout period.

            # Append the returns under the criterion and the action probabilities to the logs.
            payoffs_log1.append([episode, run, returns[0]])
            payoffs_log2.append([episode, run, returns[1]])
            act_hist_log[0].append([episode, run, action_hist1[0], action_hist1[1], action_hist1[2]])
            act_hist_log[1].append([episode, run, action_hist2[0], action_hist2[1], action_hist2[2]])

            # If we are in the last 10% of episodes we build up a state distribution log.
            if episode >= 0.9 * episodes:
                state_dist_log = update_state_dist(selected_actions, state_dist_log)

    end = time.time()
    elapsed_mins = (end - start) / 60.0
    print("Minutes elapsed: " + str(elapsed_mins))

    return payoffs_log1, payoffs_log2, act_hist_log, state_dist_log


def create_data_dir(criterion, game, opt_init, rand_prob):
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


def save_data(path, payoffs_log1, payoffs_log2, act_hist_log, state_dist_log, runs, episodes, communicate):
    info = 'NE_'

    if communicate:
        info += 'comm'
    else:
        info += 'no_comm'

    columns = ['Episode', 'Trial', 'Payoff']
    df1 = pd.DataFrame(payoffs_log1, columns=columns)
    df2 = pd.DataFrame(payoffs_log2, columns=columns)

    df1.to_csv(f'{path}/agent1_{info}.csv', index=False)
    df2.to_csv(f'{path}/agent2_{info}.csv', index=False)

    columns = ['Episode', 'Trial', 'Action 1', 'Action 2', 'Action 3']
    df1 = pd.DataFrame(act_hist_log[0], columns=columns)
    df2 = pd.DataFrame(act_hist_log[1], columns=columns)

    df1.to_csv(f'{path}/agent1_probs_{info}.csv', index=False)
    df2.to_csv(f'{path}/agent2_probs_{info}.csv', index=False)

    state_dist_log /= runs * (0.1 * episodes)
    df = pd.DataFrame(state_dist_log)
    df.to_csv(f'{path}/states_{info}.csv', index=False, header=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-game', type=str, default='game1', choices=['game1', 'game2', 'game3', 'game4', 'game5'],
                        help="which MONFG game to play")
    parser.add_argument('-criterion', type=str, default='SER', choices=['SER', 'ESR'],
                        help="optimization criterion to use")
    parser.add_argument('-communicate', action='store_true', help="Allow communication")

    parser.add_argument('-runs', type=int, default=100, help="number of trials")
    parser.add_argument('-episodes', type=int, default=5000, help="number of episodes")
    parser.add_argument('-rollout', type=int, default=100, help="Rollout period to test the current strategies")

    # Optimistic initialization can encourage exploration.
    parser.add_argument('-opt_init', action='store_true', help="optimistic initialization")
    parser.add_argument('-rand_prob', action='store_true', help="rand init for optimization prob")

    args = parser.parse_args()

    # Extracting the arguments.
    game = args.game
    criterion = args.criterion
    communicate = args.communicate
    runs = args.runs
    episodes = args.episodes
    rollout = args.rollout
    opt_init = args.opt_init
    rand_prob = args.rand_prob

    # Starting the experiments.
    payoff_matrix = get_payoff_matrix(game)
    data = run_experiment(runs, episodes, rollout, criterion, communicate, payoff_matrix, opt_init, rand_prob)
    payoffs_log1, payoffs_log2, act_hist_log, state_dist_log = data

    # Writing the data to disk.
    path = create_data_dir(criterion, game, communicate)
    save_data(path, payoffs_log1, payoffs_log2, act_hist_log, state_dist_log, runs, episodes, communicate)
