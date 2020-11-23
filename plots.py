import pandas as pd
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import seaborn as sns
from utils import *

sns.set()
sns.despine()
sns.set_context("paper", rc={"font.size": 18, "axes.labelsize": 18, "xtick.labelsize": 16, "ytick.labelsize": 16,
                             "legend.fontsize": 16})
sns.set_style('white', {'axes.edgecolor': "0.5", "pdf.fonttype": 42})
plt.gcf().subplots_adjust(bottom=0.15)

provide_comms = False
criterion = 'SER'
rand_prob = False
episodes = 5000

for game in ['game1', 'game2', 'game3', 'game4', 'game5']:
    for opt_init in [False]:  # [True, False]:
        path_data = f'data/{criterion}/{game}'
        path_plots = f'plots/{criterion}/{game}'

        if opt_init:
            path_data += '/opt_init'
            path_plots += '/opt_init'
        else:
            path_data += '/zero_init'
            path_plots += '/zero_init'

        if rand_prob:
            path_data += '/opt_rand'
            path_plots += '/opt_rand'
        else:
            path_data += '/opt_eq'
            path_plots += '/opt_eq'

        print("Creating plot path: " + repr(path_plots))
        mkdir_p(path_plots)

        if provide_comms:
            info = 'comms'
        else:
            info = 'no_comms'

        df1 = pd.read_csv(f'{path_data}/agent1_{info}.csv')
        df2 = pd.read_csv(f'{path_data}/agent2_{info}.csv')
        df1 = df1.iloc[::5, :]
        df2 = df2.iloc[::5, :]

        ax = sns.lineplot(x='Episode', y='Payoff', linewidth=2.0, data=df1, ci='sd', label='Agent 1')
        ax = sns.lineplot(x='Episode', y='Payoff', linewidth=2.0, data=df2, ci='sd', label='Agent 2')
        ax.set(ylabel='Scalarised payoff')
        ax.set_ylim(1, 18)
        ax.set_xlim(0, episodes)
        plot_name = f"{path_plots}/{game}_{criterion}_{info}"

        plt.savefig(plot_name + ".pdf")
        plt.clf()

        # Plot the action probabilities for Agent 1
        df1 = pd.read_csv(f'{path_data}/agent1_probs_{info}.csv')
        df1 = df1.iloc[::5, :]

        if game == 'game2':
            label2 = 'R'
        else:
            label2 = 'M'

        ax = sns.lineplot(x='Episode', y='Action 1', linewidth=2.0, data=df1, ci='sd', label='L')
        ax = sns.lineplot(x='Episode', y='Action 2', linewidth=2.0, data=df1, ci='sd', label=label2)

        if game in ['game1', 'game5']:
            ax = sns.lineplot(x='Episode', y='Action 3', linewidth=2.0, data=df1, ci='sd', label='R')
        ax.set(ylabel='Action probability')
        # if provide_recs:
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, episodes)
        plot_name = f"{path_plots}/{game}_{criterion}_A1_{info}"

        plt.savefig(plot_name + ".pdf")
        plt.clf()

        # Plot the action probabilities for Agent 2
        df1 = pd.read_csv(f'{path_data}/agent2_probs_{info}.csv')
        df1 = df1.iloc[::5, :]

        ax = sns.lineplot(x='Episode', y='Action 1', linewidth=2.0, data=df1, ci='sd', label='L')
        ax = sns.lineplot(x='Episode', y='Action 2', linewidth=2.0, data=df1, ci='sd', label=label2)
        if game in ['game1', 'game5']:
            ax = sns.lineplot(x='Episode', y='Action 3', linewidth=2.0, data=df1, ci='sd', label='R')
        ax.set(ylabel='Action probability')
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, episodes)
        plot_name = f"{path_plots}/{game}_{criterion}_A2_{info}"

        plt.savefig(plot_name + ".pdf")
        plt.clf()

        # Plot distribution over the joint action space, for the last 1k episodes, over all trials
        df = pd.read_csv(f'{path_data}/states_{info}.csv', header=None)

        if game == 'game2':
            x_axis_labels = ["L", "R"]
            y_axis_labels = ["L", "R"]

        if game in ['game3', 'game4']:
            x_axis_labels = ["L", "M"]
            y_axis_labels = ["L", "M"]

        if game in ['game1', 'game5']:
            x_axis_labels = ["L", "M", "R"]
            y_axis_labels = ["L", "M", "R"]

        ax = sns.heatmap(df, annot=True, cmap="YlGnBu", vmin=0, vmax=1, xticklabels=x_axis_labels,
                         yticklabels=y_axis_labels)
        plot_name = f"{path_plots}/{game}_{criterion}_states_{info}"

        plt.savefig(plot_name + ".pdf")
        plt.clf()
