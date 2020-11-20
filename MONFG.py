import time
import argparse
import pandas as pd
from utils import *
from games import *
from QLearnerESR import QLearnerESR, calc_utility
from QLearnerSER import QLearnerSER
from collections import Counter


def select_actions():
    """
    This function will select actions from the starting message for each agent
    :return: The actions that were selected.
    """
    global communicator, message
    selected = []
    for ag in range(num_agents):
        selected.append(agents[ag].select_action_mixed_nonlinear(message))
    return selected


def calc_payoffs():
    """
    This function will calculate the payoffs of the agents.
    :return: /
    """
    global payoffs
    payoffs.clear()
    for ag in range(num_agents):
        payoffs.append(payoff_matrix[selected_actions[0]][selected_actions[1]])  # Append the payoffs from the actions.


def decay_params():
    """
    Decay the parameters of the Q-learning algorithm.
    :return: /
    """
    global alpha, epsilon
    alpha *= alpha_decay
    epsilon *= epsilon_decay
    for ag in range(num_agents):
        agents[ag].alpha = alpha
        agents[ag].epsilon = epsilon


def update():
    """
    This function gets called after every episode to update the Q-tables.
    :return: /
    """
    for ag in range(num_agents):
        agents[ag].update_q_table(message, selected_actions[ag], payoffs[ag])
        agents[ag].update_joint_table(selected_actions, payoffs[ag])


def get_message(ep):
    """
    This function prepares the communication for this episode. This is the preferred action of a specific agent.
    :param ep: The current episode.
    :return: The preferred joint action.
    """
    global communicator, message
    communicator = ep % num_agents
    if 1:  # TODO: Let the agent decide when it wishes to communicate something
        message = agents[communicator].pref_joint_action()
    else:
        message = None
    return communicator, message


def do_episode(ep):
    """
    Runs an entire episode of the game.
    :param ep: The current episode.
    :return: /
    """
    global selected_actions, payoffs, current_states
    if provide_comms:
        get_message(ep)
    selected_actions = select_actions()
    calc_payoffs()
    update()
    decay_params()


def reset(opt=False, rand_prob=False):
    """
    This function will reset all variables for the new episode.
    :param opt: Boolean that decides on optimistic initialization of the Q-tables.
    :param rand_prob: Boolean that decides on random initialization for the mixed  strategy.
    :return: /
    """
    global communicator, message, current_states, selected_actions, alpha, epsilon
    agents.clear()
    for ag in range(num_agents):
        if criterion == 'SER':
            new_agent = QLearnerSER(ag, alpha, gamma, epsilon, num_states, num_actions, num_objectives, opt, rand_prob)
        else:
            new_agent = QLearnerESR(ag, alpha, gamma, epsilon, num_states, num_actions, num_objectives, opt, rand_prob)
        agents.append(new_agent)
    communicator = -1
    message = 0
    current_states = [0, 0]
    selected_actions = [-1, -1]
    alpha = alpha_start
    epsilon = epsilon_start


def run_experiment(runs, episodes, payoff_matrix, num_agents, num_objectives):
    start = time.time()

    for run in range(runs):
        print("Starting run ", r)
        reset(opt_init, rand_prob)
        action_hist = [[] for ag in range(num_agents)]
        for episode in range(episodes):
            do_episode(episode)
            payoffs = []
            actions = []
            for r in range(rollout):
                actions, payoffs = do_episode(episode)

            payoff_episode_log1.append([e, r, calc_utility(0, payoffs[0])])
            payoff_episode_log2.append([e, r, calc_utility(1, payoffs[1])])
            for i in range(num_agents):
                action_hist[i].append(selected_actions[i])
            if e >= 0.9 * num_episodes:
                state_distribution_log[selected_actions[0], selected_actions[1]] += 1

            payoffs = []

        # transform action history into probabilities
        for a, el in enumerate(action_hist):
            for i in range(len(el)):
                if i + window < len(el):
                    count = Counter(el[i:i + window])
                else:
                    count = Counter(el[-window:])
                total = sum(count.values())
                act_probs = [0, 0, 0]
                for action in range(num_actions):
                    act_probs[action] = count[action] / total
                act_hist_log[a].append([i, r, act_probs[0], act_probs[1], act_probs[2]])

    end = time.time()
    elapsed_mins = (end - start) / 60.0
    print("Minutes elapsed: " + str(elapsed_mins))


def save_data():
        path_data = f'data/{criterion}/{game}'

        if opt_init:
            path_data += '/opt_init'
        else:
            path_data += '/zero_init'

        if rand_prob:
            path_data += '/opt_rand'
        else:
            path_data += '/opt_eq'

        print("Creating data path: " + repr(path_data))
        mkdir_p(path_data)

        info = 'NE_'

        if provide_comms:
            info += 'comm'
        else:
            info += 'no_comm'

        columns = ['Episode', 'Trial', 'Payoff']
        df1 = pd.DataFrame(payoff_episode_log1, columns=columns)
        df2 = pd.DataFrame(payoff_episode_log2, columns=columns)

        df1.to_csv(f'{path_data}/agent1_{info}.csv', index=False)
        df2.to_csv(f'{path_data}/agent2_{info}.csv', index=False)

        columns = ['Episode', 'Trial', 'Action 1', 'Action 2', 'Action 3']
        df1 = pd.DataFrame(act_hist_log[0], columns=columns)
        df2 = pd.DataFrame(act_hist_log[1], columns=columns)

        df1.to_csv(f'{path_data}/agent1_probs_{info}.csv', index=False)
        df2.to_csv(f'{path_data}/agent2_probs_{info}.csv', index=False)

        state_distribution_log /= num_runs * (0.1 * num_episodes)
        df = pd.DataFrame(state_distribution_log)
        df.to_csv(f'{path_data}/states_{info}.csv', index=False, header=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-game', type=str, default='game1', choices=['game1', 'game2', 'game3', 'game4', 'game5'],
                        help="which MONFG game to play")
    parser.add_argument('-criterion', type=str, default='SER', choices=['SER', 'ESR'],
                        help="optimization criterion to use")
    parser.add_argument('-communicate', action='store_true', help="Allow communication")

    parser.add_argument('-runs', type=int, default=100, help="number of trials")
    parser.add_argument('-episodes', type=int, default=10000, help="number of episodes")
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
    data = run_experiment(runs, episodes, rollout, payoff_matrix)

    # Writing the data to disk.
    save_data()
    communicator = -1
    message = 0
    num_objectives = 2
    num_agents = 2
    num_actions = payoff_matrix.shape[0]
    num_states = num_actions ** num_agents  # Number of possible joint action messages
    agents = []
    selected_actions = [-1, -1]
    payoffs = [-1, -1]
    current_states = [0, 0]
    alpha = 0.05
    alpha_start = 0.05
    alpha_decay = 1
    epsilon = 0.1
    epsilon_start = 0.1
    epsilon_decay = 0.999
    gamma = 0.0  # this should always be zero as a MONFG is a stateless one shot decision problem
    payoff_log = []

    payoff_episode_log1 = []
    payoff_episode_log2 = []
    state_distribution_log = np.zeros((num_actions, num_actions))
    action_hist = [[], []]
    act_hist_log = [[], []]
    window = 100
    final_policy_log = [[], []]


