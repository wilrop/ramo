import argparse
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from utils import *

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sns.set()
sns.despine()
sns.set_context("paper", rc={"font.size": 18, "axes.labelsize": 18, "xtick.labelsize": 16, "ytick.labelsize": 16,
                             "legend.fontsize": 16})
sns.set_style('white', {'axes.edgecolor': "0.5", "pdf.fonttype": 42})
plt.gcf().subplots_adjust(bottom=0.15)


def plot_returns(path_plots, game, criterion, name, ag1_data, ag2_data):
    """
    This function will plot the returns obtained by both agents in one graph.
    :param path_plots: The path to which we save the plot.
    :param game: The game that was played.
    :param criterion: The multi-objective optimisation criterion that was used.
    :param name: The name of the experiment.
    :param ag1_data: The data for agent 1.
    :param ag2_data: The data for agent 2.
    :return: /
    """
    print("Plotting expected returns under: " + repr(criterion))
    ax = sns.lineplot(x='Episode', y='Payoff', linewidth=2.0, data=ag1_data, ci='sd', label='Agent 1')
    ax = sns.lineplot(x='Episode', y='Payoff', linewidth=2.0, data=ag2_data, ci='sd', label='Agent 2')
    if criterion == 'SER':
        ax.set(ylabel='Scalarised Expected Returns')
    else:
        ax.set(ylabel='Expected Scalarised Returns')

    ax.set_ylim(1, 18)
    ax.set_xlim(0, episodes)

    plot_name = f"{path_plots}/{game}_{criterion}_{name}"

    plt.savefig(plot_name + ".pdf")
    plt.clf()
    print("Finished plotting expected returns under: " + repr(criterion))


def plot_action_probabilities(path_plots, game, episodes, agent, data):
    """
    This function will plot the action probabilities for a given agent.
    :param path_plots: The path to which we save the plot.
    :param game: The game that was played.
    :param episodes: The number of episodes that was ran.
    :param agent: The agent that is being plotted.
    :param data: The data for this agent.
    :return:
    """
    print("Plotting action probabilities for agent: " + repr(agent))
    if game == 'game2':
        label2 = 'R'
    else:
        label2 = 'M'

    ax = sns.lineplot(x='Episode', y='Action 1', linewidth=2.0, data=data, ci='sd', label='L')
    ax = sns.lineplot(x='Episode', y='Action 2', linewidth=2.0, data=data, ci='sd', label=label2)

    if game in ['game1', 'game5']:
        ax = sns.lineplot(x='Episode', y='Action 3', linewidth=2.0, data=data, ci='sd', label='R')

    ax.set(ylabel='Action probability')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, episodes)
    plot_name = f"{path_plots}/{game}_{criterion}_{agent}_{name}"

    plt.savefig(plot_name + ".pdf")
    plt.clf()
    print("Finished plotting action probabilities for agent: " + repr(agent))


def plot_state_distribution(path_plots, game, data):
    """
    This function will plot the state distribution as a heatmap.
    :param path_plots: The path to which we save the plot.
    :param game: The game that was played.
    :param data: The state distribution data.
    :return: /
    """
    print("Plotting the state distribution.")
    if game == 'game2':
        x_axis_labels = ["L", "R"]
        y_axis_labels = ["L", "R"]
    elif game in ['game3', 'game4']:
        x_axis_labels = ["L", "M"]
        y_axis_labels = ["L", "M"]
    elif game in ['game1', 'game5']:
        x_axis_labels = ["L", "M", "R"]
        y_axis_labels = ["L", "M", "R"]
    else:
        raise Exception("The provided game does not exist.")

    ax = sns.heatmap(data, annot=True, cmap="YlGnBu", vmin=0, vmax=1, xticklabels=x_axis_labels,
                     yticklabels=y_axis_labels)
    plot_name = f"{path_plots}/{game}_{criterion}_states_{name}"

    plt.savefig(plot_name + ".pdf")
    plt.clf()
    print("Finished plotting the state distribution.")


def plot_results(games, criterion, name, episodes, opt_init, rand_prob):
    """
    This function will call the different plotting functions that we need for each requested game.
    :param games: The games we want to plot the results for.
    :param criterion: The multi-objective optimisation criterion that was used.
    :param name: The name of the experiment.
    :param episodes: The amount of episodes that was ran.
    :param opt_init: Whether optimistic initialization was used
    :param rand_prob: Whether rand init for optimization prob was used.
    :return:
    """
    for game in games:
        print("Generating plots for: " + repr(game))
        # Get all the paths in order.
        path_data = create_game_path('data', criterion, game, opt_init, rand_prob)
        path_plots = create_game_path('plots', criterion, game, opt_init, rand_prob)
        mkdir_p(path_plots)

        # Plot the returns for both actions in one plot.
        df1 = pd.read_csv(f'{path_data}/agent1_{name}.csv')
        df2 = pd.read_csv(f'{path_data}/agent2_{name}.csv')
        df1 = df1.iloc[::5, :]
        df2 = df2.iloc[::5, :]
        plot_returns(path_plots, game, criterion, name, df1, df2)

        # Plot the action probabilities for both agents in a separate plot.
        df1 = pd.read_csv(f'{path_data}/agent1_probs_{name}.csv')
        df1 = df1.iloc[::5, :]
        df2 = pd.read_csv(f'{path_data}/agent2_probs_{name}.csv')
        df2 = df2.iloc[::5, :]
        plot_action_probabilities(path_plots, game, episodes, 'A1', df1)
        plot_action_probabilities(path_plots, game, episodes, 'A2', df2)

        # Plot the state distribution.
        df = pd.read_csv(f'{path_data}/states_{name}.csv', header=None)
        plot_state_distribution(path_plots, game, df)
        print("Finished generating plots for: " + repr(game))
        print("------------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-games', type=str, default=['game1', 'game2', 'game3', 'game4', 'game5'], nargs='+',
                        help="Which games to plot results for.")
    parser.add_argument('-criterion', type=str, default='SER', choices=['SER', 'ESR'],
                        help="Optimisation criterion that was used.")

    parser.add_argument('-name', type=str, default='no_comms', help='The name of the experiments.')
    parser.add_argument('-episodes', type=int, default=5000, help="The number of episodes that were ran.")

    # Optimistic initialization can encourage exploration.
    parser.add_argument('-opt_init', action='store_true', help="Whether optimistic initialization was used.")
    parser.add_argument('-rand_prob', action='store_true', help="Whether rand init for optimization prob was used.")

    args = parser.parse_args()

    # Extracting the arguments.
    games = args.games
    criterion = args.criterion
    name = args.name
    episodes = args.episodes
    opt_init = args.opt_init
    rand_prob = args.rand_prob

    # Plotting the results for the requested games
    plot_results(games, criterion, name, episodes, opt_init, rand_prob)