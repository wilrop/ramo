import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import ramo.utils.experiments as ue

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sns.set()
sns.despine()
sns.set_context("paper", rc={"font.size": 18, "axes.labelsize": 18, "xtick.labelsize": 16, "ytick.labelsize": 16,
                             "legend.fontsize": 16})
sns.set_style('white', {'axes.edgecolor': "0.5", "pdf.fonttype": 42})
plt.gcf().subplots_adjust(bottom=0.15)


def save_plot(plot_name, filetype, dpi=300):
    """Save a generated plot with the requested name and filetype.

    Args:
        plot_name (str): The file to save the plot to.
        filetype (str): The filetype for the figure.
        dpi (int, optional): The resolution for the final figure. Used when saving as png. (Default value = 300)

    Returns:

    """
    if filetype == 'png':
        plt.savefig(plot_name + ".png", dpi=dpi)
    else:
        plt.savefig(plot_name + ".pdf")

    plt.clf()


def plot_returns(path_plots, filetype, game, name, ag1_data, ag2_data, min_x=0, max_x=5000, min_y=0, max_y=20):
    """Plot the returns obtained by both agents in one graph.

    Args:
        path_plots (str): The path for saving the plot.
        filetype (str): The filetype to save the file under.
        game (str): The game that was played.
        name (str): The name of the experiment that was ran.
        ag1_data (DataFrame): The data for agent 1.
        ag2_data (DataFrame): The data for agent 2.
        min_x (float, optional): The minimum value for the x-axis. (Default value = 0)
        max_x (float, optional): The maximum value for the x-axis. (Default value = 5000)
        min_y (float, optional): The minimum value for the y-axis. (Default value = 0)
        max_y (float, optional): The maximum value for the y-axis. (Default value = 20)

    Returns:

    """
    print(f'Plotting scalarised expected returns')
    ax = sns.lineplot(x='Episode', y='Payoff', linewidth=2.0, data=ag1_data, ci='sd', label='Leader')
    ax = sns.lineplot(x='Episode', y='Payoff', linewidth=2.0, data=ag2_data, ci='sd', label='Follower')
    if game == 'game7':
        x_data = np.arange(0, 5000, 5)
        y_constant = np.full(len(x_data), 0)
        ax = sns.lineplot(x=x_data, y=y_constant, linewidth=2.0, linestyle='--', label='Lower bound', color='grey')

    ax.set(ylabel='Scalarised Expected Returns')
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    plot_name = f"{path_plots}/{name}_{game}_returns"
    plt.tight_layout()
    save_plot(plot_name, filetype)

    print(f'Finished plotting scalarised expected returns')


def plot_action_probabilities(path_plots, filetype, game, name, agent, data, min_x=0, max_x=5000, min_y=0, max_y=20):
    """Plot the action probabilities for a given agent.

    Args:
        path_plots (str): The path for saving the plot.
        filetype (str): The filetype to save the file under.
        game (str): The game that was played.
        name (str): The name of the experiment that was ran.
        agent (str): The agent that is being plotted.
        data (DataFrame): The data for this agent.
        min_x (float, optional): The minimum value for the x-axis. (Default value = 0)
        max_x (float, optional): The maximum value for the x-axis. (Default value = 5000)
        min_y (float, optional): The minimum value for the y-axis. (Default value = 0)
        max_y (float, optional): The maximum value for the y-axis. (Default value = 20)

    Returns:

    """
    print(f'Plotting action probabilities for agent: {agent}')
    if game == 'game6':
        label1 = "Dare"
        label2 = "Chicken"
    elif game == 'game2':
        label1 = 'L'
        label2 = 'R'
    else:
        label1 = 'L'
        label2 = 'M'

    ax = sns.lineplot(x='Episode', y='Action 1', linewidth=2.0, data=data, ci='sd', label=label1)
    ax = sns.lineplot(x='Episode', y='Action 2', linewidth=2.0, data=data, ci='sd', label=label2)

    if game in ['game1', 'game5']:
        ax = sns.lineplot(x='Episode', y='Action 3', linewidth=2.0, data=data, ci='sd', label='R')

    ax.set(ylabel='Action probability')
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    plot_name = f"{path_plots}/{name}_{game}_{agent}_probs"
    plt.tight_layout()
    save_plot(plot_name, filetype)

    print(f'Finished plotting action probabilities for agent: {agent}')


def plot_joint_action_distribution(path_plots, game, name, data):
    """Plot a heatmap of the joint-action distribution.

    Args:
        path_plots (str): The path for saving the plot.
        game (str): The game that was played.
        name (str): The name of the experiment that was ran.
        data (DataFrame): The state distribution data.

    Returns:

    """
    print(f'Plotting the state distribution')
    if game in ['game2', 'game6', 'game8', 'game9', 'game10', 'game11']:
        x_axis_labels = ["L", "R"]
        y_axis_labels = ["L", "R"]
    elif game in ['game3', 'game4']:
        x_axis_labels = ["L", "M"]
        y_axis_labels = ["L", "M"]
    elif game in ['game1', 'game5']:
        x_axis_labels = ["L", "M", "R"]
        y_axis_labels = ["L", "M", "R"]
    elif game in ['game7']:
        x_axis_labels = ["Dare", "Chicken"]
        y_axis_labels = ["Dare", "Chicken"]
    else:
        raise Exception("The provided game does not exist.")

    if game == 'game5':
        ax = sns.heatmap(data, annot=True, cmap="YlGnBu", vmin=0, vmax=1, xticklabels=x_axis_labels,
                         yticklabels=y_axis_labels, cbar=True)
    else:
        ax = sns.heatmap(data, annot=True, cmap="YlGnBu", vmin=0, vmax=1, xticklabels=x_axis_labels,
                         yticklabels=y_axis_labels, cbar=False)

    plot_name = f"{path_plots}/{name}_{game}_states"
    plt.tight_layout()
    save_plot(plot_name, filetype)

    print(f'Finished plotting the state distribution')


def plot_com_probabilities(path_plots, filetype, game, name, agent, data, min_x=0, max_x=5000, min_y=0, max_y=20):
    """Plot the commitment or communication probabilities for a given agent.

    Args:
        path_plots (str): The path for saving the plot.
        filetype (str): The filetype to save the file under.
        game (str): The game that was played.
        name (str): The name of the experiment that was ran.
        agent (str): The agent that is being plotted.
        data (DataFrame): The data for this agent.
        min_x (float, optional): The minimum value for the x-axis. (Default value = 0)
        max_x (float, optional): The maximum value for the x-axis. (Default value = 5000)
        min_y (float, optional): The minimum value for the y-axis. (Default value = 0)
        max_y (float, optional): The maximum value for the y-axis. (Default value = 20)

    Returns:

    """
    print(f'Plotting commitment/communication probabilities for agent: {agent}')

    ax = sns.lineplot(x='Episode', y='No communication', linewidth=2.0, data=data, ci='sd', label='No Communication')
    ax = sns.lineplot(x='Episode', y='Communication', linewidth=2.0, data=data, ci='sd', label='Communication')

    ax.set(ylabel='Communication probability')
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    plot_name = f"{path_plots}/{name}_{game}_{agent}_com"
    plt.tight_layout()
    save_plot(plot_name, filetype)

    print(f'Finished plotting commitment/communication probabilities for agent: {agent}')


def plot_results(games, name, parent_dir=None, filetype='pdf'):
    """Executes the different plotting functions needed for each requested game.

    Args:
        games (List[str]): The games to plot the results for.
        name (str): The name of the experiment.
        parent_dir (str, optional): Parent directory for data and plots. (Default value = None)
        filetype (str): The filetype to save the file under. (Default value = 'pdf')

    Returns:

    """
    for game in games:
        print(f'Generating plots for: {game}')
        # Get all the paths in order.
        path_data = ue.create_game_path('data', name, game, parent_dir=parent_dir, mkdir=False)
        path_plots = ue.create_game_path('plots', name, game, parent_dir=parent_dir, mkdir=True)

        # Read returns data from the CSV files.
        df1 = pd.read_csv(f'{path_data}/{name}_{game}_A1_returns.csv')
        df2 = pd.read_csv(f'{path_data}/{name}_{game}_A2_returns.csv')

        # Define the ranges for the x and y axis.
        min_x = df1['Episode'].min()
        max_x = df1['Episode'].max() + 1  # Episodes start at 0
        min_y = 0
        max_payoff = df1['Payoff'].max()
        max_y = max_payoff - (max_payoff % 10) + 10

        # Plot the returns for both actions in one plot.
        df1 = df1.iloc[::5, :]
        df2 = df2.iloc[::5, :]
        plot_returns(path_plots, filetype, game, name, df1, df2, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y)

        # Plot the action probabilities for both agents in a separate plot.
        df1 = pd.read_csv(f'{path_data}/{name}_{game}_A1_probs.csv')
        df2 = pd.read_csv(f'{path_data}/{name}_{game}_A2_probs.csv')
        df1 = df1.iloc[::5, :]
        df2 = df2.iloc[::5, :]
        min_y = 0
        max_y = 1
        plot_action_probabilities(path_plots, filetype, game, name, 'A1', df1, min_x=min_x, max_x=max_x, min_y=min_y,
                                  max_y=max_y)
        plot_action_probabilities(path_plots, filetype, game, name, 'A2', df2, min_x=min_x, max_x=max_x, min_y=min_y,
                                  max_y=max_y)

        # Plot the state distribution.
        df = pd.read_csv(f'{path_data}/{name}_{game}_states.csv', header=None)
        plot_joint_action_distribution(path_plots, game, name, df)

        # Plot the message probabilities if applicable.
        if name in ['opt_comp_action', 'opt_coop_action', 'opt_coop_policy']:
            df1 = pd.read_csv(f'{path_data}/{name}_{game}_A1_com.csv')
            df1 = df1.iloc[::5, :]
            df2 = pd.read_csv(f'{path_data}/{name}_{game}_A2_com.csv')
            df2 = df2.iloc[::5, :]
            plot_com_probabilities(path_plots, filetype, game, name, 'A1', df1, min_x=min_x, max_x=max_x, min_y=min_y,
                                   max_y=max_y)
            plot_com_probabilities(path_plots, filetype, game, name, 'A2', df2, min_x=min_x, max_x=max_x, min_y=min_y,
                                   max_y=max_y)

        print(f'Finished generating plots for: {game}')
        print('------------------------------------------------')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--games', type=str, default=['game7'], nargs='+', help="Which games to plot results for.")
    parser.add_argument('--experiment', type=str, default='best_response', help='The experiment to generate plots for.')
    parser.add_argument('--dir', type=str, default='/Users/willemropke/Documents/mo-game-theory',
                        help='Parent directory for data and plots.')
    parser.add_argument('--filetype', type=str, default='pdf', help="The filetype to save the plots under.")

    args = parser.parse_args()

    # Extracting the arguments.
    games = args.games
    experiment = args.experiment
    parent_dir = args.dir
    filetype = args.filetype

    # Plotting the results for the requested games
    plot_results(games, experiment, parent_dir=parent_dir, filetype=filetype)
